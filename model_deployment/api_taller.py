from flask import Flask
from flask_restx import Api, Resource, reqparse
from m_model_deployment import predict_popularity

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title='Predicción de Popularidad en Canciones',
    description='API que predice la popularidad de canciones basándose en características de audio',
    mask=None  # Solo esto, sin doc='/swagger'
)

ns = api.namespace('predict', description='Predicción de la popularidad en canciones')

parser = reqparse.RequestParser(trim=True)
parser.add_argument('duration_ms', type=float, required=True,
                    help='Duración de la canción en milisegundos (133860 a 5237295)', location='args')
parser.add_argument('acousticness', type=float, required=True,
                    help='Grado de acústica (0.0 a 0.996)', location='args')
parser.add_argument('valence', type=float, required=True,
                    help='Medida de positividad musical (0.0 a 0.995)', location='args')
parser.add_argument('speechiness', type=float, required=True,
                    help='Cantidad de voz hablada en la pista (0.0 a 0.965)', location='args')
parser.add_argument('danceability', type=float, required=True,
                    help='Grado de bailable de la canción (0.0 a 0.985)', location='args')

@ns.route('/')
class PopularityApi(Resource):
    @api.expect(parser)
    def get(self):
        args = parser.parse_args()
        try:
            result = predict_popularity(**args)
            return {
                "predicted_popularity_score": result["predicted_popularity_score"],
                "popularity_category": result["popularity_category"]
            }, 200
        except ValueError as e:
            return {'message': str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
