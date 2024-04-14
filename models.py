from configuration.config import *
from peewee import *

connection = PostgresqlDatabase(db_name,
    user=user,
    password=password,
    host=host,
    port=port)


class BaseModel(Model):
    class Meta:
        database = connection

class Camera(BaseModel):

    id = PrimaryKeyField()
    name  = TextField()
    date = DateTimeField()
    picture = BlobField()
    


    class Meta:
        db_table = 'Camera'
        order_by = ('id',)