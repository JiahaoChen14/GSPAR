# Third-Party Software Notices

GSPAR is based on third-party frameworks and components. The MIT licenses for NerfICG and the original GSPAR contributions do not alter the license terms that apply to the components below or their dependencies. Before publishing or redistributing this repository, verify the copyright notices, license texts, and dependency list against the files and submodule revisions actually included in the distribution.

| Project or component | Use in this project | License information and location |
| --- | --- | --- |
| [NerfICG](https://github.com/nerficg-project/nerficg) | Base framework for training, dataset processing, and visualization | The base framework is licensed under the MIT License. Retain the upstream copyright notice and the repository-root `LICENSE`. Individual methods and extensions may be subject to additional license terms. |
| [nvblox](https://github.com/nvidia-isaac/nvblox) | Depth-derived TSDF and ESDF maps used as experimental references | Review and retain `GSPAR/nvblox_gpu_sdf/NVBLOX_LICENSE`, and verify the provenance and license of any source code copied or modified from nvblox. |
| [JPS3D](https://github.com/KumarRobotics/jps3d) | Optional JPS planning backend | Review and retain `GSPAR/path/jps3d/LICENSE`. |
| [stdgpu](https://github.com/stotko/stdgpu) | Dependency of the nvblox reference-map workflow | Review and retain `GSPAR/nvblox_gpu_sdf/third_party/stdgpu/LICENSE`. |
| NerfICG methods and extensions | 3DGS training and other reconstruction functionality | Review the original licenses in `src/Methods/`, `src/Thirdparty/`, and each submodule individually; for example, see `src/Methods/HTGS_egsr/LICENSE`. |

SplatNav is referenced as an external experimental baseline. If its source code is not included in this repository, it is not a bundled software component. Any other third-party source code, model weights, datasets, or submodules included in a distribution remain subject to their respective license terms.

**Important:** The upstream NerfICG documentation states that its base framework uses the MIT License, but some optional Gaussian Splatting extensions and the default accelerated implementation are subject to non-commercial licenses. Neither the repository-root MIT License nor the license for the original GSPAR contributions replaces those component-specific terms. Before publishing or redistributing the project, review the code, revisions, and original license files actually included in the distribution.
