## Api para stock prediction

Estando en este directorio (/api) correr los siguientes comandos:

```
docker build -t stock-predictor-api .
docker run -p 8000:8000 stock-predictor-api
```

Verificar que la api esta corriendo accediendo a:

```
http://localhost:8000/docs
```
