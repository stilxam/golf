{
  description = "Nix Flake for Piecewise Linear Fitting in Jax";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python312;

        cudatoolkit = pkgs.cudatoolkit;

        cudaEnvHook = ''
          export CUDA_HOME=${cudatoolkit}
          export CUDA_ROOT=${cudatoolkit}
          export LD_LIBRARY_PATH="${cudatoolkit.lib}/lib:${cudatoolkit}/lib:$LD_LIBRARY_PATH"
          export PATH="${cudatoolkit}/bin:$PATH"
          export CMAKE_PREFIX_PATH="${cudatoolkit}:$CMAKE_PREFIX_PATH"
        '';

        golf-src = pkgs.fetchFromGitHub {
          owner = "stilxam";
          repo = "golf";
          rev = "e95cacccd5ce8857844eb7f0db3abc72f94aaac2";
          sha256 = "sha256-UoJ7ozFxrTUKQk8FcOws/uwyue+v1xEpq4NOe3xhqXo=";
        };

        pythonPackages = pkgs.python312.pkgs;

        golf = pythonPackages.buildPythonPackage {
          pname = "golf";
          version = "2960b45"; # Using the git revision for version
          src = golf-src;
          propagatedBuildInputs = with pythonPackages; [
            jax
            jaxlib
            equinox
            jaxtyping
            optax
          ];
        };

        mainPythonPackages = ps: with ps; [
          pip
          cython
          jax
          jaxlib
          equinox
          jaxtyping
          optax
          notebook
          matplotlib
          golf
        ];

        pythonEnv = python.withPackages mainPythonPackages;

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            cudatoolkit
          ] ;

          shellHook = cudaEnvHook + ''
          '';
        };
      });
}