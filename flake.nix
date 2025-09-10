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

        plf-src = pkgs.fetchFromGitHub {
          owner = "stilxam";
          repo = "plf";
          rev = "a7ef1f3684751b80b9f3ad9d36e3e3ab7745275d";
          sha256 = "sha256-ZnbUmNI5cnvetcJoBjCsWn+nh5qMhS7pn2UlOfIm6bA=";
        };

        pythonPackages = pkgs.python312.pkgs;

        plf = pythonPackages.buildPythonPackage {
          pname = "plf";
          version = "2960b45"; # Using the git revision for version
          src = plf-src;
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
          plf
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