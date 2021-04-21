

FROM python:3.6-stretch

RUN pip3 -q install pip --upgrade

COPY . .

RUN pip3 install -r requirements.txt

RUN pip3 install jupyter

RUN apt-get install unzip

RUN wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip 

WORKDIR /notebooks 

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]




