#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <jps_planner/jps_planner/graph_search.h>

namespace {

bool read_input(
    const std::string& path,
    int& nx,
    int& ny,
    int& nz,
    int& sx,
    int& sy,
    int& sz,
    int& gx,
    int& gy,
    int& gz,
    double& eps,
    int& use_jps,
    int& max_expand,
    std::vector<char>& cmap) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "cannot open input file: " << path << "\n";
    return false;
  }

  std::string magic;
  std::getline(in, magic);
  if (magic != "NERFICG_JPS3D_V1") {
    std::cerr << "invalid input magic: " << magic << "\n";
    return false;
  }

  std::string data_token;
  in >> nx >> ny >> nz;
  in >> sx >> sy >> sz;
  in >> gx >> gy >> gz;
  in >> eps >> use_jps >> max_expand;
  in >> data_token;
  if (!in || data_token != "DATA") {
    std::cerr << "invalid input header\n";
    return false;
  }
  in.get();

  const std::int64_t total = static_cast<std::int64_t>(nx) * ny * nz;
  if (nx <= 0 || ny <= 0 || nz <= 0 || total <= 0) {
    std::cerr << "invalid dimensions\n";
    return false;
  }
  cmap.resize(static_cast<std::size_t>(total));
  in.read(cmap.data(), static_cast<std::streamsize>(cmap.size()));
  if (in.gcount() != static_cast<std::streamsize>(cmap.size())) {
    std::cerr << "input map payload is truncated\n";
    return false;
  }
  return true;
}

bool in_bounds(int x, int y, int z, int nx, int ny, int nz) {
  return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
}

std::int64_t key(int x, int y, int z, int nx, int ny) {
  return static_cast<std::int64_t>(x) + static_cast<std::int64_t>(nx) * (y + static_cast<std::int64_t>(ny) * z);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: nerficg_jps3d_cli <input.bin>\n";
    return 2;
  }

  int nx = 0, ny = 0, nz = 0;
  int sx = 0, sy = 0, sz = 0;
  int gx = 0, gy = 0, gz = 0;
  double eps = 1.0;
  int use_jps = 1;
  int max_expand = -1;
  std::vector<char> cmap;
  if (!read_input(argv[1], nx, ny, nz, sx, sy, sz, gx, gy, gz, eps, use_jps, max_expand, cmap)) {
    return 2;
  }

  if (!in_bounds(sx, sy, sz, nx, ny, nz) || !in_bounds(gx, gy, gz, nx, ny, nz)) {
    std::cout << "FAIL\nreason endpoint_out_of_bounds\n";
    return 1;
  }
  if (cmap[static_cast<std::size_t>(key(sx, sy, sz, nx, ny))] != 0 ||
      cmap[static_cast<std::size_t>(key(gx, gy, gz, nx, ny))] != 0) {
    std::cout << "FAIL\nreason endpoint_not_free\n";
    return 1;
  }

  const auto started = std::chrono::steady_clock::now();
  JPS::GraphSearch planner(cmap.data(), nx, ny, nz, eps, false);
  const bool ok = planner.plan(sx, sy, sz, gx, gy, gz, use_jps != 0, max_expand);
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  const auto closed = planner.getCloseSet();
  const auto all = planner.getAllSet();
  if (!ok) {
    std::cout << "FAIL\n";
    std::cout << "reason no_path_or_limit\n";
    std::cout << "stats " << closed.size() << " " << all.size() << " " << elapsed << "\n";
    return 1;
  }

  auto path = planner.getPath();
  std::cout << "OK\n";
  std::cout << "stats " << closed.size() << " " << all.size() << " " << elapsed << "\n";
  std::cout << "path " << path.size() << "\n";
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    std::cout << (*it)->x << " " << (*it)->y << " " << (*it)->z << "\n";
  }
  return 0;
}
