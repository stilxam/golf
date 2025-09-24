{
  description = "A nix flake for JAX+EQX to fit piecewise linear functions";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    unstable-nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, unstable-nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        unstable-pkgs = import unstable-nixpkgs {
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
          rev = "33cfd208f8314369c1e36437f93ee54bceede54b";
          sha256 = "sha256-AczJss7+uij0eqmBetYh8i9JRwVrTz+d1bSK76NmgPI=";
        };

        pythonPackages = pkgs.python312.pkgs;

        golf = pythonPackages.buildPythonPackage {
          pname = "golf";
          version = "1.0"; # Using the git revision for version
          src = golf-src;
          propagatedBuildInputs = with pythonPackages; [
            jax
            jaxlib
            equinox
            jaxtyping
            optax
          ];
        };
 #        
	# py-earth-src = pkgs.fetchFromGitHub {
 #          owner = "scikit-learn-contrib";
 #          repo = "py-earth";
 #          rev = "b209d1916f051dbea5b142af25425df2de469c5a";
 #          sha256 = "sha256-8CGnrCNwy0t618JApTwfhSx5XXLGQ2qYjoS4w/qHMsY=";
 #        };
	#
	#
 #        py-earth = pythonPackages.buildPythonPackage {
 #          pname = "py-earth";
 #          version = "0.1.0"; 
 #          src = py-earth-src;
	#   postPatch = ''
	#     sed -i 's/configparser.SafeConfigParser/configparser.ConfigParser/g' versioneer.py
	#     sed -i 's/parser.readfp(f)/parser.read_file(f)/g' versioneer.py
	#   '';
 #          propagatedBuildInputs = with pythonPackages; [
	#     scipy
	#     scikit-learn
	#     six
	#     cython
	#     pandas
	#     statsmodels
	#     sympy
	#     patsy
	#     numpy
 #          ];
 #        };
	#


        mainPythonPackages = ps: with ps; [
        cython
        jax
        jaxlib
        equinox
        jaxtyping
        optax
        einops

        notebook

        scipy
        pwlf
	# py-earth
        golf

        pandas
        matplotlib
        seaborn
        plotly
        wandb
        tqdm
        beartype
        ];

        pythonEnv = python.withPackages mainPythonPackages;
      in
      {
        devShells.default = pkgs.mkShell {
           
           
          buildInputs = [
            pythonEnv
	    cudatoolkit
	    unstable-pkgs.gemini-cli
          ] ;

          shellHook = cudaEnvHook + ''
            echo "CUDA toolkit available at: $CUDA_HOME"
            echo "Python environment with JAX, CUDA, EQX and Jaxtyping."
          '';
        };
      });
}
