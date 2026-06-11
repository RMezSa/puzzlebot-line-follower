Conversación con Gemini
Pasame esto a md file



Proyecto M5 · Control Inteligente

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

Variable de entradaUniverso de discursoFunciones de membresíaBase de reglas difusasMétodo de inferenciaDefuzzificación

Explicar en presentación

Por qué se eligieron esas funciones de membresíaQué comportamiento produce cada reglaControlador 2

Sugeno

Misma entrada y mismas dos salidas que el controlador Mamdani. El equipo debe justificar si utilizó salidas constantes, lineales u otra forma permitida dentro del enfoque Sugeno, y cómo esa decisión afecta la suavidad, interpretabilidad y desempeño.

El diseño debe incluir

Variable de entradaFunciones de membresíaFunciones de salidaBase de reglas

Explicar en presentación

Procedimiento de cálculo de salidasComparación directa con MamdaniProceso requerido

Sintonización de reglas y funciones de membresía

La sintonización es libre, pero se debe mostrar evidencia del proceso de mejora — no únicamente la versión final.

Evidencia esperada

Versiones iniciales de membresíasCambios en reglasProblemas observadosAjustes aplicadosComparación entre versión inicial y finalAnálisis

Curva de control y función matemática propuesta

Con una sola entrada, la relación entrada-salida se representa como una curva de control — no como una superficie bidimensional. Mostrar la curva del mejor controlador para cada salida.

Curvas requeridas

Entrada vs. velocidad linealEntrada vs. velocidad angular

Función matemática propuesta

A partir de la mejor respuesta obtenida, proponer una función matemática que capture el comportamiento principal de las curvas. La función no tiene que ser idéntica al controlador difuso, pero debe justificarse su similitud y sus limitaciones.

v(e) = v_max − k_v |e|

w(e) = −k_w e

w(e) = −k_w tanh(ae)   ejemplo no lineal

Requerido

Comparación entre Mamdani y Sugeno

La comparación debe ir más allá de decir cuál funcionó mejor. Las conclusiones deben estar justificadas con observaciones, gráficas, pruebas o argumentos técnicos.

ejemplos de Criterios de comparación

Facilidad de implementaicónSuavidad de la respuestaInterpretabilidadEstabilidad de seguimientoOscilacionesVelocidad promedioRecuperación ante erroresCosto computacionalITAE Parte 2 — Propuesta de mejora con segunda entrada

Agregar una segunda entrada al sistema difuso con justificación clara. La segunda entrada debe responder a una necesidad real observada en la primera parte — no debe agregarse solo para hacer el controlador más complejo.

Ejemplos de posibles segundas entradas

Cambio del error respecto al tiempoCurvatura estimada de la líneaDiferencia entre sensores lateralesError anteriorVelocidad actual del robotConfianza en la detecciónProceso requerido

Diseño, prueba y análisis con dos entradas

Pasos requeridos

Definir las dos entradas del controladorJustificar por qué la segunda entrada mejora el desempeñoDiseñar membresías para ambas entradasDiseñar reglas con interacción entre entradasProbar el desempeño del controladorComparar con la mejor versión de la Parte 1

Superficies de control requeridas (dos entradas)

Superficie de velocidad linealSuperficie de velocidad angular Fundamentos esperados

Control difuso

Qué es una variable lingüísticaQué representa una función de membresíaCómo se calcula el grado de activaciónCómo se combinan reglas difusasDiferencias entre Mamdani y SugenoCómo las reglas generan una ley de control globalSeguimiento de línea

Cómo se detecta la líneaCómo se calcula la entrada del controladorCómo se normaliza la entrada, si aplicaCómo se interpretan errores positivos y negativosCómo se decide la velocidad lineal y angularQué ocurre cuando el robot pierde la línea Entregables

La calificación se asigna únicamente con base en la presentación técnica y la sesión Q&A. No hay reporte escrito evaluado. El equipo debe tener disponible evidencia suficiente para defender su trabajo: código, bitácora de sintonización, gráficas, videos, tablas de reglas y resultados experimentales.

01

Presentación técnica

Descripción del problema y plataformaDiseño Mamdani y SugenoFunciones de membresía y reglasProceso de sintonizaciónCurvas y superficies de controlFunción matemática propuestaPropuesta con segunda entradaCasos de falla y conclusiones02

Evidencia funcional

Comportamiento del controlador MamdaniComportamiento del controlador SugenoComportamiento del controlador con dos entradasDiferencias entre versionesAl menos un caso de falla o limitación Sesión Q&A

30%

de la evaluación total

Reglas de la sesión oral

Cualquier integrante puede ser seleccionado para responder

Solo un integrante responde por pregunta

Sin intervención del resto del equipo

La calidad de cada respuesta impacta la evaluación de todo el equipo

Temas posibles

Funciones de membresíaReglas difusasMamdani vs. SugenoDefuzzificaciónActivación de reglasSuperficie de controlFunción matemática propuestaSintonizaciónComportamiento del robotSensores utilizadosCasos de fallaJustificaciones ingenieriles

Tu archivo Markdown (.md) está listo
Icono de MD
 proyecto_m...nteligente 
 MD 

He formateado todo el texto que proporcionaste para que tenga una estructura profesional y limpia en formato Markdown. En este documento he agregado:

Encabezados jerárquicos (#, ##, ###) para separar claramente cada sección del proyecto.

Listas y viñetas para hacer mucho más fácil la lectura de los entregables y las instrucciones de la presentación.

Fórmulas matemáticas utilizando la sintaxis de LaTeX ($$...$$) para que las funciones propuestas (v(e), w(e)) se rendericen correctamente.

Énfasis (negritas y cursivas) para resaltar los entregables clave y las métricas de evaluación (como el 30% del Q&A).

Puedes descargar el archivo y visualizarlo en cualquier editor compatible con Markdown (como VS Code, Obsidian, GitHub, etc.). Si necesitas ajustar alguna sección o agregar más detalles, dime y con gusto lo actualizo.




Gemini es una IA y puede cometer errores.

# Proyecto M5 · Control Inteligente
## Controlador Difuso Seguidor de Línea
### Mamdani · Sugeno · Mejora con segunda entrada
*Diseño, sintonización, análisis y comparación de controladores difusos para seguimiento de línea*

---

## 🎯 Objetivo
Implementar, sintonizar, analizar y comparar controladores difusos para un robot móvil seguidor de línea. El proyecto busca que el estudiante comprenda cómo una estrategia basada en reglas lingüísticas puede generar comandos de movimiento para un robot, y cómo dichas reglas pueden justificarse bajo criterios ingenieriles.

El enfoque principal no es únicamente lograr que el robot siga una línea, sino demostrar una comprensión profunda del diseño del controlador, sus reglas, sus funciones de membresía, sus salidas y sus limitaciones.

---

## 📋 Descripción general del problema
El robot debe seguir una línea usando un sistema de control difuso. La entrada del controlador debe representar el estado del robot respecto a la línea — por ejemplo, el error lateral, el error normalizado de sensores o la posición relativa de la línea.

### Salidas del controlador
* **Salida 1 (Linear velocity):** Velocidad lineal del robot.
* **Salida 2 (Angular velocity):** Velocidad angular del robot.

---

## ⚙️ Parte 1 — Una entrada, dos salidas

### 1. Controlador 1: Mamdani
Controlador difuso tipo Mamdani con una entrada y dos salidas. Cada decisión de diseño debe poder justificarse técnica y conceptualmente.

**El diseño debe incluir:**
* Variable de entrada
* Universo de discurso
* Funciones de membresía
* Base de reglas difusas
* Método de inferencia
* Defuzzificación

**Explicar en la presentación:**
* Por qué se eligieron esas funciones de membresía.
* Qué comportamiento produce cada regla.

### 2. Controlador 2: Sugeno
Misma entrada y mismas dos salidas que el controlador Mamdani. El equipo debe justificar si utilizó salidas constantes, lineales u otra forma permitida dentro del enfoque Sugeno, y cómo esa decisión afecta la suavidad, interpretabilidad y desempeño.

**El diseño debe incluir:**
* Variable de entrada
* Funciones de membresía
* Funciones de salida
* Base de reglas

**Explicar en la presentación:**
* Procedimiento de cálculo de salidas.
* Comparación directa con Mamdani.

---

### 🔄 Proceso requerido

#### Sintonización de reglas y funciones de membresía
La sintonización es libre, pero se debe mostrar evidencia del proceso de mejora — no únicamente la versión final.

**Evidencia esperada:**
* Versiones iniciales de membresías.
* Cambios en reglas.
* Problemas observados.
* Ajustes aplicados.
* Comparación entre versión inicial y final.

---

### 📊 Análisis

#### Curva de control y función matemática propuesta
Con una sola entrada, la relación entrada-salida se representa como una curva de control (no como una superficie bidimensional). Mostrar la curva del mejor controlador para cada salida.

**Curvas requeridas:**
* Entrada vs. velocidad lineal
* Entrada vs. velocidad angular

#### Función matemática propuesta
A partir de la mejor respuesta obtenida, proponer una función matemática que capture el comportamiento principal de las curvas. La función no tiene que ser idéntica al controlador difuso, pero debe justificarse su similitud y sus limitaciones.

* **Velocidad lineal:** $$v(e) = v_{max} - k_v |e|$$
* **Velocidad angular (Lineal):** $$w(e) = -k_w e$$
* **Velocidad angular (Ejemplo no lineal):** $$w(e) = -k_w \tanh(ae)$$

---

### ⚖️ Comparación requerida entre Mamdani y Sugeno
La comparación debe ir más allá de decir cuál funcionó mejor. Las conclusiones deben estar justificadas con observaciones, gráficas, pruebas o argumentos técnicos.

**Criterios de comparación sugeridos:**
* Facilidad de implementación
* Suavidad de la respuesta
* Interpretabilidad
* Estabilidad de seguimiento
* Oscilaciones
* Velocidad promedio
* Recuperación ante errores
* Costo computacional
* ITAE (Integral del Tiempo por el Valor Absoluto del Error)

---

## 🚀 Parte 2 — Propuesta de mejora con segunda entrada

Agregar una segunda entrada al sistema difuso con justificación clara. La segunda entrada debe responder a una necesidad real observada en la primera parte — no debe agregarse solo para hacer el controlador más complejo.

**Ejemplos de posibles segundas entradas:**
* Cambio del error respecto al tiempo ($de/dt$)
* Curvatura estimada de la línea
* Diferencia entre sensores laterales
* Error anterior
* Velocidad actual del robot
* Confianza en la detección

### 🛠️ Proceso requerido: Diseño, prueba y análisis con dos entradas
1.  Definir las dos entradas del controlador.
2.  Justificar por qué la segunda entrada mejora el desempeño.
3.  Diseñar membresías para ambas entradas.
4.  Diseñar reglas con interacción entre entradas.
5.  Probar el desempeño del controlador.
6.  Comparar con la mejor versión de la Parte 1.

**Superficies de control requeridas (dos entradas):**
* Superficie de velocidad lineal
* Superficie de velocidad angular

---

## 📚 Fundamentos esperados

### Control difuso
* Qué es una variable lingüística.
* Qué representa una función de membresía.
* Cómo se calcula el grado de activación.
* Cómo se combinan reglas difusas.
* Diferencias entre Mamdani y Sugeno.
* Cómo las reglas generan una ley de control global.

### Seguimiento de línea
* Cómo se detecta la línea.
* Cómo se calcula la entrada del controlador.
* Cómo se normaliza la entrada, si aplica.
* Cómo se interpretan errores positivos y negativos.
* Cómo se decide la velocidad lineal y angular.
* Qué ocurre cuando el robot pierde la línea.

---

## 📦 Entregables

La calificación se asigna únicamente con base en la presentación técnica y la sesión Q&A. **No hay reporte escrito evaluado.** El equipo debe tener disponible evidencia suficiente para defender su trabajo: código, bitácora de sintonización, gráficas, videos, tablas de reglas y resultados experimentales.

### 01. Presentación técnica
* Descripción del problema y plataforma.
* Diseño Mamdani y Sugeno.
* Funciones de membresía y reglas.
* Proceso de sintonización.
* Curvas y superficies de control.
* Función matemática propuesta.
* Propuesta con segunda entrada.
* Casos de falla y conclusiones.

### 02. Evidencia funcional
* Comportamiento del controlador Mamdani.
* Comportamiento del controlador Sugeno.
* Comportamiento del controlador con dos entradas.
* Diferencias entre versiones.
* Al menos un caso de falla o limitación.

---

## ❓ Sesión Q&A (30% de la evaluación total)

### Reglas de la sesión oral:
* Cualquier integrante puede ser seleccionado para responder.
* Solo un integrante responde por pregunta (sin intervención del resto del equipo).
* La calidad de cada respuesta impacta la evaluación de todo el equipo.

### Temas posibles:
* Funciones de membresía y reglas difusas.
* Mamdani vs. Sugeno.
* Defuzzificación y activación de reglas.
* Superficie de control y función matemática propuesta.
* Sintonización y comportamiento del robot.
* Sensores utilizados y casos de falla.
* Justificaciones ingenieriles.