# development shell, includes all dependencies, debug, style tools
{
  pkgs ? import <nixpkgs> {}
}:
with pkgs;
pkgs.mkShell {
        name = "miniqp";
	nativeBuildInputs = [];
        buildInputs = [
          # dependencies for the code
          python3
          python3Packages.black
          python3Packages.notebook
          python3Packages.matplotlib
          python3Packages.ipywidgets
          python3Packages.tqdm
          python3Packages.numpy
          python3Packages.mypy
          python3Packages.flake8
        ];
}
