<<<<<<< HEAD
## Integrantes

* a1721 Dimas Ignacio Torres (dimaciodimacio@gmail.com)
* a1726	Joaquín Matías Mestanza (joa.mestanza@gmail.com)
* a1714	Ramiro Andrés Feichubuinm (ra.feichu@gmail.com)
=======
## Autores

- Dimas Ignacio Torres
- Joaquín Matías Mestanza
- Ramiro Andrés Feichubuinm
>>>>>>> b5073e78419cd281b7b8277d8e4f55d13c2c24b9

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
```

Realizar una nueva run:

```
docker-compose restart yahoo_trainer
```

Para ver log de errores de script:

```
docker-compose logs yahoo_trainer
```

Después de haber hecho un cambio en el script de python:

```
docker-compose build yahoo_trainer
docker-compose up -d yahoo_trainer
```

## Pendientes

- Automatizar flujos de trabajo con Airflow

- Crear una API del modelo con Flask
