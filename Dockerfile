FROM python:3.7


# establishing the work dir
WORKDIR /app

# upgrading pip
RUN python -m pip install --upgrade pip


# installing the requirements
COPY ./requirements.txt .
RUN pip install -r  requirements.txt

# exposing the port
EXPOSE 8501

# copy all other files
COPY . .

# # Creating an entrypoint
ENTRYPOINT ["streamlit"]

# # running the end command
CMD ["run", "fin.py"]