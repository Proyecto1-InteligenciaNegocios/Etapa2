from pydantic import BaseModel


class DataModel(BaseModel):
    text: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "La investigación de homicidios es una de las tareas más prestigiosas y difíciles de los cuerpos de seguridad modernos. La mayoría de los departamentos de policía metropolitanos..."
                },
                {
                    "text": "El capítulo examina la contribución que la teoría de las Relaciones Internacionales ha hecho a la lectura y práctica de la construcción de paz."
                }
            ]
        }
    }