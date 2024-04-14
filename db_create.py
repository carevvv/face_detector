from models import *
from peewee import *

with connection:
    connection.create_tables([Camera])

print("Done")