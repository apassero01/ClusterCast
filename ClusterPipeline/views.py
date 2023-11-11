from django.shortcuts import render
import json
from django.http import JsonResponse
from .models import SequencePreprocessing as SP
from .models import ClusterProcessing as CP
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import plotly.graph_objects as go
import plotly


# Create your views here.
@csrf_exempt
def home(request):
    # Only allow POST requests for this view
    if request.method == 'POST':
        # Load the JSON data from the request body
        data = json.loads(request.body)
        print("req")
        # Unpack the dictionary and do something with it
        tickersString = data.get('tickers')
        tickers = tickersString.split(',')
        tickers = [ticker.strip() for ticker in tickers]
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        steps = int(data.get('steps'))
        interval = (data.get('interval'))
        cluster_features_string = data.get('cluster_features')
        cluster_features = cluster_features_string.split(',')
        cluster_features = [feature.strip() for feature in cluster_features]
        cluster_features = [feature.replace("'","") for feature in cluster_features]
        print(cluster_features)

        target_features_string = data.get('target_features')
        target_features = target_features_string.split(',')
        target_features = [feature.strip() for feature in target_features]
        target_features = [feature.replace("'","") for feature in target_features]

        print(target_features)

        # Process the data (this is where you would include your logic)
        group_params = CP.StockClusterGroupParams(tickers = tickers, start_date = start_date, end_date = end_date, n_steps = steps, cluster_features = cluster_features, target_cols = target_features, interval=interval)
        cluster_group = CP.StockClusterGroup(group_params)

        cluster_group.create_data_set() 
        cluster_group.create_sequence_set()

        X_train, y_train, X_test, y_test = cluster_group.get_3d_array()
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        cluster_group.run_clustering()
        cluster_group.create_clusters()

        cluster_graphs = [] 
    
        for cluster in cluster_group.clusters:
            fig1 = cluster.visualize_cluster()
            fig1_json = json.loads(plotly.io.to_json(fig1))
            fig2 = cluster.visualize_target_values() 
            fig2_json = json.loads(plotly.io.to_json(fig2))
            cluster_graphs.append((fig1_json,fig2_json))

        pct_Chg_cols = next(filter(lambda feature_set: feature_set.name == 'pctChg_vars', cluster_group.group_params.X_feature_sets)).cols
        cuma_cols = next((filter(lambda feature_set: "cum" in feature_set.name, cluster_group.group_params.X_feature_sets))).cols
        price_cols = next((filter(lambda feature_set: "price" in feature_set.name, cluster_group.group_params.X_feature_sets))).cols

        rolling_cols = [] 
        rolling_features = list((filter(lambda feature_set: "rolling" in feature_set.name, cluster_group.group_params.X_feature_sets)))
        for feature in rolling_features: 
            rolling_cols += feature.cols

        train_cols = cuma_cols
        cluster_group.train_all_rnns(train_cols)
        # print(cluster_graphs)
        # Return a JSON response with a success message or the processed data
        return JsonResponse({'figures': cluster_graphs})

    # If it's a GET request, just render the page as usual
    return render(request, 'ClusterPipeline/home.html')

