{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.setuptools
    pkgs.python3Packages.wheel
    pkgs.python3Packages.fastapi
    pkgs.python3Packages.uvicorn
    pkgs.python3Packages.pydantic
    pkgs.python3Packages.email-validator
    pkgs.python3Packages.transformers
    pkgs.python3Packages.torch
    pkgs.python3Packages.numpy
    pkgs.python3Packages.pandas
    pkgs.python3Packages.scipy
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.pillow
    pkgs.nodejs-16_x
    pkgs.nodePackages.npm
    pkgs.yarn
    pkgs.git
    pkgs.curl
  ];
  env = {
    PYTHONBIN = "${pkgs.python3}/bin/python3";
    LANG = "en_US.UTF-8";
    PYTHONUNBUFFERED = "1";
    PIP_ROOT_USER_ACTION = "ignore";
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ];
  };
}