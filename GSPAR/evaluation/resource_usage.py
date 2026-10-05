"""Consistent CPU RSS and PyTorch CUDA memory measurements for evaluations."""

from __future__ import annotations

import os
import resource
import sys
import threading
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


def current_rss_bytes() -> int:
    """Return current process RSS without requiring psutil."""

    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as file:
            resident_pages = int(file.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024


@dataclass(frozen=True)
class ResourceUsage:
    cpu_rss_baseline_bytes: int
    cpu_rss_peak_sampled_bytes: int
    cpu_rss_peak_sampled_delta_bytes: int
    cpu_rss_end_bytes: int
    cpu_sample_interval_ms: float
    cuda_monitor_enabled: bool
    cuda_memory_available: bool
    cuda_device: str | None
    cuda_monitor_error: str | None
    cuda_torch_allocated_baseline_bytes: int | None
    cuda_torch_allocated_peak_bytes: int | None
    cuda_torch_allocated_peak_delta_bytes: int | None
    cuda_torch_reserved_baseline_bytes: int | None
    cuda_torch_reserved_peak_bytes: int | None
    cuda_torch_reserved_peak_delta_bytes: int | None

    def to_dict(self) -> dict[str, int | float | bool | str | None]:
        return asdict(self)


class ResourceMonitor:
    """Measure process RSS and PyTorch allocator peaks for one operation.

    RSS is sampled in a background thread, so a peak shorter than the sampling
    interval can be missed. CUDA values cover the PyTorch caching allocator;
    driver and non-PyTorch device allocations are not included.
    """

    def __init__(
        self,
        *,
        cpu_sample_interval_sec: float = 0.01,
        monitor_cuda: bool = True,
        cuda_device: str | int | None = None,
    ) -> None:
        self.cpu_sample_interval_sec = max(float(cpu_sample_interval_sec), 0.001)
        self.monitor_cuda = bool(monitor_cuda) and (
            cuda_device is None
            or isinstance(cuda_device, int)
            or str(cuda_device).startswith("cuda")
        )
        self.cuda_device = cuda_device
        self.usage: ResourceUsage | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_baseline = 0
        self._cpu_peak = 0
        self._torch = None
        self._device = None
        self._cuda_available = False
        self._cuda_monitor_error: str | None = None
        self._cuda_allocated_baseline: int | None = None
        self._cuda_reserved_baseline: int | None = None

    def _prepare_cuda(self) -> None:
        if not self.monitor_cuda:
            return
        try:
            import torch

            self._cuda_available = bool(torch.cuda.is_available())
            if not self._cuda_available:
                return
            requested = (
                torch.device("cuda", int(self.cuda_device))
                if isinstance(self.cuda_device, int)
                else torch.device(self.cuda_device)
                if self.cuda_device is not None
                else torch.device("cuda", torch.cuda.current_device())
            )
            device_index = (
                torch.cuda.current_device()
                if requested.index is None
                else int(requested.index)
            )
            self._torch = torch
            self._device = torch.device("cuda", device_index)
            torch.cuda.synchronize(self._device)
            self._cuda_allocated_baseline = int(
                torch.cuda.memory_allocated(self._device)
            )
            self._cuda_reserved_baseline = int(
                torch.cuda.memory_reserved(self._device)
            )
            torch.cuda.reset_peak_memory_stats(self._device)
        except Exception as error:  # monitoring must not mask evaluation errors
            self._cuda_monitor_error = f"{type(error).__name__}: {error}"
            self._torch = None
            self._device = None

    def _sample_cpu(self) -> None:
        while not self._stop.wait(self.cpu_sample_interval_sec):
            self._cpu_peak = max(self._cpu_peak, current_rss_bytes())

    def __enter__(self) -> "ResourceMonitor":
        # Importing torch can itself use substantial host memory. Prepare the
        # monitor before taking the evaluation baseline so that overhead is not
        # attributed to the operation being measured.
        self._prepare_cuda()
        self._cpu_baseline = current_rss_bytes()
        self._cpu_peak = self._cpu_baseline
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_cpu,
            name="evaluation-rss-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        cpu_end = current_rss_bytes()
        self._cpu_peak = max(self._cpu_peak, cpu_end)

        allocated_peak = None
        reserved_peak = None
        if self._torch is not None:
            try:
                self._torch.cuda.synchronize(self._device)
                allocated_peak = int(
                    self._torch.cuda.max_memory_allocated(self._device)
                )
                reserved_peak = int(
                    self._torch.cuda.max_memory_reserved(self._device)
                )
            except Exception as error:  # monitoring remains best-effort
                self._cuda_monitor_error = f"{type(error).__name__}: {error}"

        self.usage = ResourceUsage(
            cpu_rss_baseline_bytes=self._cpu_baseline,
            cpu_rss_peak_sampled_bytes=self._cpu_peak,
            cpu_rss_peak_sampled_delta_bytes=max(
                0, self._cpu_peak - self._cpu_baseline
            ),
            cpu_rss_end_bytes=cpu_end,
            cpu_sample_interval_ms=1000.0 * self.cpu_sample_interval_sec,
            cuda_monitor_enabled=self.monitor_cuda,
            cuda_memory_available=self._cuda_available,
            cuda_device=None if self._device is None else str(self._device),
            cuda_monitor_error=self._cuda_monitor_error,
            cuda_torch_allocated_baseline_bytes=self._cuda_allocated_baseline,
            cuda_torch_allocated_peak_bytes=allocated_peak,
            cuda_torch_allocated_peak_delta_bytes=(
                None
                if allocated_peak is None or self._cuda_allocated_baseline is None
                else max(0, allocated_peak - self._cuda_allocated_baseline)
            ),
            cuda_torch_reserved_baseline_bytes=self._cuda_reserved_baseline,
            cuda_torch_reserved_peak_bytes=reserved_peak,
            cuda_torch_reserved_peak_delta_bytes=(
                None
                if reserved_peak is None or self._cuda_reserved_baseline is None
                else max(0, reserved_peak - self._cuda_reserved_baseline)
            ),
        )

    def to_dict(self) -> dict[str, int | float | bool | str | None]:
        if self.usage is None:
            raise RuntimeError("ResourceMonitor has not finished")
        return self.usage.to_dict()


def summarize_resource_usages(
    usages: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the maximum absolute and incremental peaks across named stages."""

    items = list(usages.items()) if isinstance(usages, Mapping) else [
        (str(index), usage) for index, usage in enumerate(usages)
    ]

    def maximum(field: str) -> tuple[int | None, str | None]:
        candidates = [
            (int(usage[field]), name)
            for name, usage in items
            if isinstance(usage.get(field), (int, float))
            and not isinstance(usage.get(field), bool)
        ]
        return max(candidates, default=(None, None))

    cpu_peak, cpu_stage = maximum("cpu_rss_peak_sampled_bytes")
    cpu_delta, cpu_delta_stage = maximum("cpu_rss_peak_sampled_delta_bytes")
    allocated_peak, allocated_stage = maximum(
        "cuda_torch_allocated_peak_bytes"
    )
    allocated_delta, allocated_delta_stage = maximum(
        "cuda_torch_allocated_peak_delta_bytes"
    )
    reserved_peak, reserved_stage = maximum("cuda_torch_reserved_peak_bytes")
    reserved_delta, reserved_delta_stage = maximum(
        "cuda_torch_reserved_peak_delta_bytes"
    )
    return {
        "stage_count": len(items),
        "peak_cpu_rss_sampled_bytes": cpu_peak,
        "peak_cpu_rss_stage": cpu_stage,
        "peak_cpu_rss_sampled_delta_bytes": cpu_delta,
        "peak_cpu_rss_delta_stage": cpu_delta_stage,
        "peak_cuda_torch_allocated_bytes": allocated_peak,
        "peak_cuda_torch_allocated_stage": allocated_stage,
        "peak_cuda_torch_allocated_delta_bytes": allocated_delta,
        "peak_cuda_torch_allocated_delta_stage": allocated_delta_stage,
        "peak_cuda_torch_reserved_bytes": reserved_peak,
        "peak_cuda_torch_reserved_stage": reserved_stage,
        "peak_cuda_torch_reserved_delta_bytes": reserved_delta,
        "peak_cuda_torch_reserved_delta_stage": reserved_delta_stage,
        "cuda_memory_available": any(
            bool(usage.get("cuda_memory_available")) for _, usage in items
        ),
    }
