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
          rev = "4e666cf90e1385a052180612a02f2908e0d588e0";
          sha256 = "sha256-gxTFhMMSeHNqB2EP6Y27ON2tQysZhc9+a+0VTz3pXL4=";
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