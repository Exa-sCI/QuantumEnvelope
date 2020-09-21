{ stdenv, pkgconfig, autoreconfHook, python3Packages, python3, src }:

python3Packages.buildPythonPackage {
      format = "other";
      name = "mnemo-py";
      src = src;
      nativeBuildInputs = [ autoreconfHook pkgconfig ];
      buildInputs = [ python3 ];
      buildPhase = "make";
      installPhase = "make install";
}
