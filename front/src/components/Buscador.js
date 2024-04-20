import React, { Component } from 'react';

class Buscador extends Component {
    busquedaRef = React.createRef();

    obtenerDatos = (e) => {
         e.preventDefault();
         const comentario = this.busquedaRef.current.value;
         this.props.datosBusqueda(comentario);
         console.log(comentario);
    }

    limpiarTexto = () => {
        // Limpiar el valor del textarea
        this.busquedaRef.current.value = '';
        // Disparar el evento 'input' para que el textarea refleje el cambio
        this.busquedaRef.current.dispatchEvent(new Event('input'));
    }

    render() {
        return (
            <form onSubmit={this.obtenerDatos} className="Buscador-form">
                <div className="row">
                    <div className='form-group col-md-8'>
                        <textarea
                            ref={this.busquedaRef}
                            className='form-control form-control-lg'
                            placeholder="Ingresa tu comentario"
                            rows="1" // Esto establece el número inicial de filas
                            style={{ minHeight: '40px' }} // Altura mínima
                            onChange={() => {
                                const textarea = this.busquedaRef.current;
                                textarea.style.height = 'auto'; // Restablecer altura a auto
                                textarea.style.height = `${textarea.scrollHeight}px`; // Ajustar altura según el contenido
                            }}
                        ></textarea>
                    </div>
                    <div className="form-group col-md-4">
                        <input type="submit" className="btn btn-lg btn-info btn-block" value="Clasificar" />
                        <button type="button" className="btn btn-lg btn-secondary btn-block" onClick={this.limpiarTexto}>Realizar otra reseña</button>
                    </div>
                </div>
            </form>
        );
    }
}

export default Buscador;


