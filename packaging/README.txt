Fabric Stretch

fabric-viewer       Open a measurement project or the included examples/demo.
fabsim              Pipeline commands; run fabsim --help.
fabsim-rig          Acquisition app. Use the simulated camera/controller to try it.
press_snapshot     Render a Press View to PNG without a window; run with --help.

The portable archive needs no Python installation. Extract the whole directory.
On Linux, use Ubuntu 22.04+ or another distribution with glibc 2.35+ and GTK 3,
Vulkan/OpenGL, X11/Wayland libraries. The .deb installs these dependencies.
The MSI adds Start Menu shortcuts and the installation directory to PATH.

The IDS uEye SDK/driver is installed separately for the real camera. The Python
bindings are included. CoWTracker and its weights are not included.

Copy examples/demo to a writable folder before rebuilding its derived files.
Use the demo's README for its tiling arguments.

Documentation: https://github.com/Personne-admin/fabric-stretch
Licence: LICENSE. Third-party notices: NOTICE and licenses/.
