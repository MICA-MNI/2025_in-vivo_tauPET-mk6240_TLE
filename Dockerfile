#----    Rstudio    ----#
# https://arca-dpss.github.io/manual-open-science/rocker-chapter.html
# Rstudio
FROM rocker/rstudio:4.4

#---- System dependencies ----#
RUN apt-get update && apt-get install -y \
    curl \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libglpk-dev \
    libgmp3-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

#---- renv ----#
ENV RENV_VERSION 0.15.1
ENV RENV_PATHS_CACHE /home/rstudio/.cache/R/renv
RUN R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
RUN R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"

#---- Install R packages ----#
RUN R -e "install.packages(c( \
    'lubridate', \
    'dplyr', \
    'gtsummary', \
    'kableExtra', \
    'scales', \
    'ggplot2', \
    'lme4', \
    'lmerTest', \
    'pander', \
    'osfr', \
    'gridExtra', \
    'emmeans', \
    'corrplot', \
    'RColorBrewer', \
    'GGally', \
    'lavaan', \
    'lavaanPlot', \
    'semPlot' \
), repos='https://cloud.r-project.org')"

# Change ownership
RUN chown -R rstudio:rstudio /home/rstudio/

#----    Environment    ----#
# Copy our R script to the container
COPY ./scripts/Increased_in-vivo_tau_in_TLE.Rmd /home/rstudio/Rscripts

# Set the working directory
WORKDIR /home/rstudio