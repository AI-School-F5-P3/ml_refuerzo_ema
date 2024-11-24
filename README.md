# ml_refuerzo_ema
Proyecto Telecomunicaciones para optimizar campañas de marketing

# Acerca del conjunto de datos

Imaginemos que un proveedor de telecomunicaciones ha segmentado su base de clientes según patrones de uso de servicios, categorizando a los clientes en cuatro grupos. Si los datos demográficos pueden predecir la pertenencia a un grupo, la empresa podría personalizar ofertas para clientes potenciales.

Este es un problema de clasificación, donde, dado un conjunto de datos con etiquetas predefinidas, necesitamos construir un modelo que prediga la clase de un caso nuevo o desconocido.

El ejemplo se centra en usar datos demográficos, como región, edad y estado civil, para predecir patrones de uso.

La variable objetivo, llamada custcat, tiene cuatro valores posibles que corresponden a los cuatro grupos de clientes:

Servicio Básico
Servicio Electrónico
Servicio Plus
Servicio Total
Nuestro objetivo es construir un clasificador para predecir la clase de casos desconocidos. Utilizaremos un tipo específico de clasificación llamado K-Nearest Neighbors (KNN).