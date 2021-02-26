{ stdenv, pkgconfig, autoreconfHook, python3Packages, python3, src, mnemo }:

python3Packages.buildPythonPackage {
      format = "other";
      name = "mnemo-py";
      src = src;
      nativeBuildInputs = [ autoreconfHook pkgconfig ];
      buildInputs = [ python3 ];
      propagatedBuildInputs = [ mnemo ];
      buildPhase = "make";
      installPhase = "make install";
}
