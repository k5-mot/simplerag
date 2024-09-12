FROM python:3.12

# Non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV PATH=/home/${USERNAME}/.local/bin:${PATH}

# Add non-root user
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME
RUN apt-get update
RUN apt-get install -y sudo
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get clean

# Update package lists
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
    curl git vim less gcc build-essential
# For Python
RUN apt-get install -y \
    libmagic-dev poppler-utils libpoppler-dev \
    tesseract-ocr libtesseract-dev tesseract-ocr-jpn \
    tesseract-ocr-jpn-vert tesseract-ocr-script-jpan \
    tesseract-ocr-script-jpan-vert \
    libxml2-dev libxslt1-dev libgl1-mesa-dev
RUN apt-get autoremove -y
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Setup git branch in console
RUN echo "source /usr/share/bash-completion/completions/git" >> /home/${USERNAME}/.bashrc
WORKDIR /usr/share/bash-completion/completions
RUN curl -O https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh
RUN curl -O https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash
RUN chmod a+x git*.*
RUN ls -l $PWD/git*.* | awk '{print "source "$9}' >> /home/${USERNAME}/.bashrc
RUN echo "GIT_PS1_SHOWUNTRACKEDFILES=true" >> /home/${USERNAME}/.bashrc
RUN echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[01;33m\] \w \[\033[01;31m\]\$(__git_ps1 \"(%s)\") \\n\[\033[01;34m\]\\$ \[\033[00m\]"' >> /home/${USERNAME}/.bashrc

# Set the working directory in the container
WORKDIR /workspace
ENV GEM_HOME=/home/USERNAME/.gems
ENV PATH=${GEM_HOME}/bin:${PATH}
COPY ./requirements.txt /workspace
RUN chown -R $USERNAME:$USERNAME /workspace

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir black flake8 isort
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade \
    langchain-anthropic langchain-unstructured \
    unstructured-client unstructured \
    "unstructured[all-docs]" python-magic \
    flashrank

# Run any command to initialize the container
EXPOSE 8000
ENTRYPOINT ["sleep", "infinity"]
# ENTRYPOINT ["chainlit", "run", "app.py"]
