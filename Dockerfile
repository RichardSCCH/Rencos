FROM coady/pylucene
WORKDIR /usr/src/app
COPY . .
RUN pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
CMD "/bin/bash"