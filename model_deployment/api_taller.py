from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from m_model_deployment import predict_popularity

app = Flask(__name__)
api = Api(app, version='1.0', title='Predicción de Popularidad en Canciones',
          description='Predice la popularidad con base en características de audio')

ns = api.namespace('predict', description='Predicción popularidad en canción')

parser = ns.parser()
parser.add_argument('duration_ms', type=float, required=True, help='Duración en milisegundos', location='args')
parser.add_argument('danceability', type=float, required=True, help='Danceabilidad', location='args')
parser.add_argument('valence', type=float, required=True, help='Valencia', location='args')

resource_fields = api.model('Respuesta', {
    'result': fields.String,
})

@ns.route('/')
class PopularityApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        result = predict_popularity(**args)
        return {'result': str(result)}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)