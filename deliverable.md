Controlador Difuso Seguidor de Línea
Mamdani · Sugeno · Mejora con segunda entrada
Diseño, sintonización, análisis y comparación de controladores difusos para seguimiento de línea

Parte 1

Una entrada, dos salidas

Controladores Mamdani y Sugeno — comparación ingenieril

Parte 2

Propuesta de mejora

Segunda entrada — superficie de control y comparación

 	
Objetivo
Implementar, sintonizar, analizar y comparar controladores difusos para un robot móvil seguidor de línea. El proyecto busca que el estudiante comprenda cómo una estrategia basada en reglas lingüísticas puede generar comandos de movimiento para un robot, y cómo dichas reglas pueden justificarse bajo criterios ingenieriles.

El enfoque principal no es únicamente lograr que el robot siga una línea, sino demostrar una comprensión profunda del diseño del controlador, sus reglas, sus funciones de membresía, sus salidas y sus limitaciones.

 	
Descripción general del problema
El robot debe seguir una línea usando un sistema de control difuso. La entrada del controlador debe representar el estado del robot respecto a la línea — por ejemplo, el error lateral, el error normalizado de sensores o la posición relativa de la línea.

Salidas del controlador

Salida 1

Linear velocity

Velocidad lineal del robot

Salida 2

Angular velocity

Velocidad angular del robot

 	
Parte 1 — Una entrada, dos salidas
Controlador 1

Mamdani

Controlador difuso tipo Mamdani con una entrada y dos salidas. Cada decisión de diseño debe poder justificarse técnica y conceptualmente.

El diseño debe incluir

Variable de entrada	Universo de discurso	Funciones de membresía
Base de reglas difusas	Método de inferencia	Defuzzificación
Explicar en presentación

Por qué se eligieron esas funciones de membresía	Qué comportamiento produce cada regla
Controlador 2

Sugeno

Misma entrada y mismas dos salidas que el controlador Mamdani. El equipo debe justificar si utilizó salidas constantes, lineales u otra forma permitida dentro del enfoque Sugeno, y cómo esa decisión afecta la suavidad, interpretabilidad y desempeño.

El diseño debe incluir

Variable de entrada	Funciones de membresía	Funciones de salida	Base de reglas
Explicar en presentación

Procedimiento de cálculo de salidas	Comparación directa con Mamdani
Proceso requerido

Sintonización de reglas y funciones de membresía

La sintonización es libre, pero se debe mostrar evidencia del proceso de mejora — no únicamente la versión final.

Evidencia esperada

Versiones iniciales de membresías	Cambios en reglas	Problemas observados
Ajustes aplicados	Comparación entre versión inicial y final
Análisis

Curva de control y función matemática propuesta

Con una sola entrada, la relación entrada-salida se representa como una curva de control — no como una superficie bidimensional. Mostrar la curva del mejor controlador para cada salida.

Curvas requeridas

Entrada vs. velocidad lineal	Entrada vs. velocidad angular
Función matemática propuesta

A partir de la mejor respuesta obtenida, proponer una función matemática que capture el comportamiento principal de las curvas. La función no tiene que ser idéntica al controlador difuso, pero debe justificarse su similitud y sus limitaciones.

v(e) = v_max − k_v |e|

w(e) = −k_w e

w(e) = −k_w tanh(ae)   ejemplo no lineal

Requerido

Comparación entre Mamdani y Sugeno

La comparación debe ir más allá de decir cuál funcionó mejor. Las conclusiones deben estar justificadas con observaciones, gráficas, pruebas o argumentos técnicos.

ejemplos de Criterios de comparación

Facilidad de implementaicón	Suavidad de la respuesta	Interpretabilidad
Estabilidad de seguimiento	Oscilaciones	Velocidad promedio
Recuperación ante errores	Costo computacional	ITAE
 	
Parte 2 — Propuesta de mejora con segunda entrada
Agregar una segunda entrada al sistema difuso con justificación clara. La segunda entrada debe responder a una necesidad real observada en la primera parte — no debe agregarse solo para hacer el controlador más complejo.

Ejemplos de posibles segundas entradas

Cambio del error respecto al tiempo	Curvatura estimada de la línea	Diferencia entre sensores laterales
Error anterior	Velocidad actual del robot	Confianza en la detección
Proceso requerido

Diseño, prueba y análisis con dos entradas

Pasos requeridos

Definir las dos entradas del controlador	Justificar por qué la segunda entrada mejora el desempeño
Diseñar membresías para ambas entradas	Diseñar reglas con interacción entre entradas
Probar el desempeño del controlador	Comparar con la mejor versión de la Parte 1
Superficies de control requeridas (dos entradas)

Superficie de velocidad lineal	Superficie de velocidad angular
 	
Fundamentos esperados
Control difuso

Qué es una variable lingüística
Qué representa una función de membresía
Cómo se calcula el grado de activación
Cómo se combinan reglas difusas
Diferencias entre Mamdani y Sugeno
Cómo las reglas generan una ley de control global
Seguimiento de línea

Cómo se detecta la línea
Cómo se calcula la entrada del controlador
Cómo se normaliza la entrada, si aplica
Cómo se interpretan errores positivos y negativos
Cómo se decide la velocidad lineal y angular
Qué ocurre cuando el robot pierde la línea
 	
Entregables
La calificación se asigna únicamente con base en la presentación técnica y la sesión Q&A. No hay reporte escrito evaluado. El equipo debe tener disponible evidencia suficiente para defender su trabajo: código, bitácora de sintonización, gráficas, videos, tablas de reglas y resultados experimentales.

01

Presentación técnica

Descripción del problema y plataforma
Diseño Mamdani y Sugeno
Funciones de membresía y reglas
Proceso de sintonización
Curvas y superficies de control
Función matemática propuesta
Propuesta con segunda entrada
Casos de falla y conclusiones
02

Evidencia funcional

Comportamiento del controlador Mamdani
Comportamiento del controlador Sugeno
Comportamiento del controlador con dos entradas
Diferencias entre versiones
Al menos un caso de falla o limitación
 	
Sesión Q&A
30%

de la evaluación total

Reglas de la sesión oral

Cualquier integrante puede ser seleccionado para responder

Solo un integrante responde por pregunta

Sin intervención del resto del equipo

La calidad de cada respuesta impacta la evaluación de todo el equipo

Temas posibles

Funciones de membresía	Reglas difusas	Mamdani vs. Sugeno	Defuzzificación
Activación de reglas	Superficie de control	Función matemática propuesta	Sintonización
Comportamiento del robot	Sensores utilizados	Casos de falla	Justificaciones ingenieriles
 	
Criterios de evaluación
Criterio	Peso	%
Diseño e implementación del controlador Mamdani	
 
10
Diseño e implementación del controlador Sugeno	
 
10
Proceso de sintonización y justificación de reglas y membresías	
 
10
Análisis de curvas o superficies y función matemática propuesta	
 
10
Propuesta de mejora con segunda entrada	
 
10
Comparación ingenieril, pros, contras y limitaciones	
 
10
Calidad técnica de la presentación	
 
10
Desempeño en sesión Q&A	
 
30
