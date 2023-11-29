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
from django.db import transaction
from django.http import HttpResponse
import ast
from .thread import CreateGroupBackground


# Create your views here.
@csrf_exempt
@transaction.atomic
def home(request):
    supported_params = CP.SupportedParams.objects.get(pk=7)
    cluster_features_list = supported_params.features
    context = {
        'cluster_features_list': cluster_features_list
    }

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
        cluster_features = data.get('cluster_features')
        print(cluster_features)

        target_features = ['sumpctChgclose_1','sumpctChgclose_2','sumpctChgclose_3','sumpctChgclose_4','sumpctChgclose_5','sumpctChgclose_6']


        print(target_features)
        scaling_dict = {
            'price_vars': SP.ScalingMethod.SBSG,
            'trend_vars' : SP.ScalingMethod.SBS,
            'pctChg_vars' : SP.ScalingMethod.QUANT_MINMAX,
            'rolling_vars' : SP.ScalingMethod.QUANT_MINMAX_G,
            'target_vars' : SP.ScalingMethod.UNSCALED
        }

        # Process the data (this is where you would include your logic)
        group_params = CP.StockClusterGroupParams.objects.create(tickers = tickers, start_date = start_date, end_date = end_date, n_steps = steps, cluster_features = cluster_features, target_cols = target_features, interval=interval)
        group_params.set_scaling_dict(scaling_dict)
        group_params.initialize()
        group_params.save() 
        
        cluster_group = CP.StockClusterGroup.objects.create(group_params = group_params)

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
        trend_cols = next((filter(lambda feature_set: "trend" in feature_set.name, cluster_group.group_params.X_feature_sets))).cols


        rolling_cols = [] 
        rolling_features = list((filter(lambda feature_set: "rolling" in feature_set.name, cluster_group.group_params.X_feature_sets)))
        for feature in rolling_features: 
            rolling_cols += feature.cols
            
        col_dict = {
            "PctChgVars": pct_Chg_cols,
            "CumulativeVars": cuma_cols,
            "RollingVars": rolling_cols,
            "PriceVars": price_cols,
            "TrendVars": trend_cols
        }

        training_selected_features = data.get("training_features")
        training_features = []
        for feature in training_selected_features:
            training_features += col_dict[feature]

        cluster_group.train_all_rnns(training_features)
        # print(cluster_graphs)
        # Return a JSON response with a success message or the processed data
        return JsonResponse({'figures': cluster_graphs})

    # If it's a GET request, just render the page as usual
    return render(request, 'ClusterPipeline/home.html',context)

@csrf_exempt
@transaction.atomic
def cluster_run(request):
    supported_params = CP.SupportedParams.objects.get(pk=7)
    cluster_features_list = supported_params.features
    context = {
        'cluster_features_list': cluster_features_list
    }

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
        cluster_features = data.get('cluster_features')
        print(cluster_features)

        target_features = ['sumpctChgclose_1','sumpctChgclose_2','sumpctChgclose_3','sumpctChgclose_4','sumpctChgclose_5','sumpctChgclose_6']


        print(target_features)
        scaling_dict = {
            'price_vars': SP.ScalingMethod.SBSG,
            'trend_vars' : SP.ScalingMethod.SBS,
            'pctChg_vars' : SP.ScalingMethod.QUANT_MINMAX,
            'rolling_vars' : SP.ScalingMethod.QUANT_MINMAX_G,
            'target_vars' : SP.ScalingMethod.UNSCALED
        }

        col_dict = {
            "PctChgVars": supported_params.pct_chg_features,
            "CumulativeVars": supported_params.cuma_features,
            "RollingVars": supported_params.rolling_features,
            "PriceVars": supported_params.price_features,
            "TrendVars": supported_params.trend_features
        }

        training_selected_features = data.get("training_features")
        training_features = []
        for feature in training_selected_features:
            training_features += col_dict[feature]

        # Process the data (this is where you would include your logic)
        group_params = CP.StockClusterGroupParams.objects.create(tickers = tickers, start_date = start_date, end_date = end_date, n_steps = steps, cluster_features = cluster_features, target_cols = target_features, interval=interval, training_features = training_features)
        group_params.set_scaling_dict(scaling_dict)
        group_params.initialize()
        group_params.save() 
        
        CreateGroupBackground(group_params).start()

        

        return HttpResponse("Success")

        cluster_results = [] 
    
        for cluster in cluster_group.clusters: 
            fig1 = cluster.visualize_cluster()
            fig1_json = json.loads(plotly.io.to_json(fig1))
            metrics = cluster.generate_results()
            cluster_results.append([fig1_json,metrics])
            # cluster_results.append([metrics])

        # print(cluster_graphs)
        # Return a JSON response with a success message or the processed data
        return JsonResponse({'results': cluster_results})

    # If it's a GET request, just render the page as usual
    return render(request, 'ClusterPipeline/create_run.html',context)

@csrf_exempt
def cluster_group(request):
    cluster_group_list = CP.StockClusterGroup.objects.all()
    item_list = [(item.pk, item.group_params.name) for item in cluster_group_list]
    if request.method == 'POST':
        data = json.loads(request.body)
        group_data_string = (data['cluster_group'])
        group_data = ast.literal_eval(group_data_string)
        id = group_data[0]
        print(id)

        cluster_group = CP.StockClusterGroup.objects.get(pk=id)
        
        cluster_results = [] 
        cluster_group.load_saved_group()
        for cluster in cluster_group.clusters: 
            fig1 = cluster.visualize_cluster()
            fig2 = cluster.visualize_future_distribution()
            fig1_json = json.loads(plotly.io.to_json(fig1))
            fig2_json = json.loads(plotly.io.to_json(fig2))
            metrics = cluster.generate_results()
            cluster_results.append([(fig1_json,fig2_json),metrics])
            # print(cluster_results[-1])
        
        return JsonResponse({'results': cluster_results})
    return render(request, 'ClusterPipeline/cluster_group.html',{'menu_items': item_list})