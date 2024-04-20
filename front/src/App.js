import React, { Component } from 'react';
import './App.css';
import Buscador from './components/Buscador';

class App extends Component {
  state = {
    comentario: "Comentario ingresado",
    calificacion: "Calificación obtenida"
  }

  consultarApi = () => {
    const comentario = this.state.comentario;
    const url = 'http://localhost:8000/predict';
    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        "text": comentario
      }),
    })
      .then(response => response.json())
      .then(resultado => {
        if (resultado) {
          this.setState({ calificacion: resultado });
        } else {
          this.setState({ calificacion: 'No se pudo clasificar' });
        }
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        this.setState({ calificacion: 'Error en la solicitud' });
      });
  }

  datosBusqueda = (comentario) => {
    this.setState({ comentario }, () => {
      this.consultarApi();
    });
  }

  render() {
    return (
      <div className="container">
        <div className="jumbotron">
          <p className="lead text-center">Clasificador de comentarios</p>
          <Buscador datosBusqueda={this.datosBusqueda} />
        </div>
        <h3>Comentario realizado</h3>
        <p>{this.state.comentario}</p>
        <h3>Calificación</h3>
        <p>{this.state.calificacion}</p>
      </div>
    );
  }
}

export default App;
