{
  description = "Pyrenew-Hew Flake";

  # Nix flakes have inputs and outputs.
  # Inputs are where you put dependencies, 
  # and outputs are what you build.

  # --- Flake Inputs ---
  # Where you get nix packages and other dependencies from
  inputs = {
    # Pin nixpkgs to a specific release for reproducibility
    # Get the nixos 25.11 release, just a github repository full of nix packages
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-25.11";
  };

  # --- Flake Outputs ---
  # What you build with the inputs
  # Notice the nixpkgs input is passed into the outputs function
  outputs = { self, nixpkgs }: 
  
  let
    
    # Define the target system architecture
    system = "x86_64-linux";

    # Import nixpkgs and apply overlays
    pkgs = import nixpkgs {
      inherit system;
      # overlays = [
      #   # Override nodejs to use version 20
      #   (final: prev: {
      #     nodejs = prev.nodejs_20;
      #   })
      # ];
    };

    # --- R Package Dependencies ---
    # List all required R packages in one place for reuse
    rDeps = with pkgs.rPackages; [

      # CRAN R packages
      arrow argparser cowplot dplyr DT fable feasts forcats fs
      ggdist ggnewscale ggplot2 glue here htmltools jsonlite knitr
      latex2exp lubridate purrr readr reticulate rlang scales
      scoringutils stringr tibble tidybayes tidyr tidyselect urca
      
      # Add github R packages
      # TODO: hubdata and hubutils
      # Build the local 'forecasttools' R package from source
      (pkgs.rPackages.buildRPackage {
        name = "forecasttools";
        pname = "forecasttools";
        version = "0.0.0.9000";
        src = pkgs.fetchFromGitHub {
          owner = "cdcgov";
          repo = "forecasttools";
          rev = "1wxy43app99lqbs07as8wm2p8qvgd9vv00vdj0y9aiq8qabmx78c"; # commit hash
          sha256 = "sha256-DJ1el8IIR5U8kG0DsHdqb2N0ReVIqwP0wjSle9UgvvM="; # nix run nixpkgs#nix-prefetch-git https://github.com/cdcgov/forecasttoolsnix
        };
        # buildInputs = rDeps ++ [ pkgs.R ];
      })
    ];

  in

    {
      # --- Development Shell ---
      devShells.${system}.default = pkgs.mkShell {

          buildInputs = with pkgs; [
            # Provide an interactive R environment with all dependencies
            (rWrapper.override { packages = rDeps; })

            # Build the local 'hewr' R package from source
            (pkgs.rPackages.buildRPackage {
              name = "hewr";
              pname = "hewr";
              version = "0.0.0.9000";
              src = ./hewr;
              buildInputs = rDeps ++ [ pkgs.R ];
            })
          ];

          shellHook = ''
            echo "R dev environment loaded."
            echo "Interactive R session and built 'hewr' package are available."
            # uncomment to print all R library versions
            # Rscript -e 'ip <- installed.packages(); cat(sprintf("%-30s %s", ip[order(ip[, "Package"]), "Package"], ip[order(ip[, "Package"]), "Version"]), sep="\n")'
            # setting timezone to match Docker env
            export TZ=Etc/UTC
          '';
          
      };
    };
}