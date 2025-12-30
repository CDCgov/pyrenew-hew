{
  description = "PyRenew-HEW Nix Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; };
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs.rPackages; [

            # Python
            pkgs.uv 

            # R
            pkgs.R

            # Toolchain for compiling R packages
            pkgs.stdenv.cc
            pkgs.pkg-config

            # System libraries needed by R packages
            pkgs.curl
            pkgs.curl.dev
            # pkgs.openssl
            # pkgs.openssl.dev
            pkgs.openssl_3
            pkgs.openssl_3.dev
            pkgs.libxml2
            pkgs.libxml2.dev
            pkgs.zlib
            pkgs.zlib.dev

            # devtools
            devtools

            # hewr dependencies, minus forecasttools
            arrow
            argparser
            cowplot
            dplyr
            DT
            fable
            feasts
            forcats
            fs
            ggdist
            ggnewscale
            ggplot2
            glue
            here
            htmltools
            jsonlite
            knitr
            latex2exp
            lubridate
            purrr
            readr
            reticulate
            rlang
            scales
            scoringutils
            stringr
            tibble
            tidybayes
            tidyr
            tidyselect
            urca
        ];

        shellHook = ''
            # Ensure libcurl is available during R package builds
            # Force OpenSSL 3.x by adding its libraries to LD_LIBRARY_PATH
            # export LD_LIBRARY_PATH="${pkgs.openssl_3.out}/lib:$LD_LIBRARY_PATH"
            
            # Ensure dynamic linker uses the R-specific libraries
            export R_HOME=$(R RHOME)
            export R_LIBS_USER="$PWD/.R/library"
            mkdir -p "$R_LIBS_USER"

            # Sync Python environment
            uv sync --active
            which python && python --version

            # Set the R repos and install missing packages
            Rscript -e "options(repos = c(CRAN = 'https://p3m.dev/cran/__linux__/noble/latest'))"
            
            # Check if hubUtils is installed, then install dependencies only if necessary
            Rscript -e "if (!requireNamespace('hubUtils', quietly=TRUE)) install.packages('hubUtils')"
            Rscript -e "if (!requireNamespace('arrow', quietly=TRUE)) install.packages('arrow', type='source')"
            Rscript -e "if (!requireNamespace('hubData', quietly=TRUE)) devtools::install_github('hubverse-org/hubData')"
            Rscript -e "if (!requireNamespace('forecasttools', quietly=TRUE)) devtools::install_github('cdcgov/forecasttools')"
            Rscript -e "if (!requireNamespace('hewr', quietly=TRUE)) devtools::install_local(path='hewr')"
        '';
    };
  };
}