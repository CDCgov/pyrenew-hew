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
            pkgs.R
            arrow
            argparser
            cowplot
            dplyr
            DT
            fable
            feasts
            forcats
            # forecasttools (>= 0.1.6)
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
            # scoringutils (>= 2.0.0)
            stringr
            tibble
            tidybayes
            tidyr
            tidyselect
            urca
        ];
    };
  };
}