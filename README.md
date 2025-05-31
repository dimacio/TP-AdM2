## Como correr el proyecto

Estando en root correr:
```
docker-compose up --build
```

Para acceder a la UI de MLFlow:
```
http://localhost:5001
```

Para acceder a Bucket (MinIO):
```
http://localhost:9001
````

Realizar una nueva run: 
```
docker-compose restart yahoo_trainer
```

Para ver log de errores de script: 
```
docker-compose logs yahoo_trainer
```
