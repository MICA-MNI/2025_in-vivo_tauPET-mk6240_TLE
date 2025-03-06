#----    Rstudio    ----#
# https://arca-dpss.github.io/manual-open-science/rocker-chapter.html
# Rstudio
FROM rocker/rstudio:4.4

# Install external dependencies
RUN apt-get update \
    && apt-get install -y curl

# Copy project files
COPY . /home/rstudio/my-project

ENV RENV_VERSION 0.15.1
ENV RENV_PATHS_CACHE /home/rstudio/.cache/R/renv
RUN R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
RUN R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"

# Install R dependencies
RUN R -e "install.packages(c('lubridate', 'dplyr', 'gtsummary', 'kableExtra', 'scales', 'ggplot2', 'reticulate', 'lme4', 'lmerTest'))"

# Change ownership
RUN chown -R rstudio:rstudio /home/rstudio/

#----    Environment    ----#
# Copy our R script to the container
COPY ./data /home/rstudio/data 
COPY ./Rscripts /home/rstudio/Rscripts

# Set the working directory
WORKDIR /home/rstudio