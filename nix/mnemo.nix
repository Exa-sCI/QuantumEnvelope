{ stdenv, pkgconfig, fetchgit, autoreconfHook, python3, src }:

stdenv.mkDerivation {
      name = "mnemo";
      src = src;
      nativeBuildInputs = [ autoreconfHook pkgconfig ];
      buildInputs = [ python3 ];
}
