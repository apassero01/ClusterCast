from django.shortcuts import render, get_object_or_404, redirect
import json
from django.http import JsonResponse
from .models import SequencePreprocessing as SP
from .models import ClusterProcessing as CP
from .models import Predictions as Pred
from .models import RNNModels as RNN
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import plotly.graph_objects as go
import plotly
import os
import pandas as pd
from django.db import transaction
from django.http import HttpResponse
import ast
from .thread import CreateGroupBackground
from django.core.cache import cache
from datetime import date, datetime, timedelta
import yfinance as yf

# Create your views here.
@csrf_exempt
@transaction.atomic
def home(request):
    return render(request, 'ClusterPipeline/home.html')

@csrf_exempt
@transaction.atomic
def cluster_run(request):
    supported_params = CP.SupportedParams.objects.get(pk=1)
    cluster_features_list = supported_params.features
    context = {"cluster_features_list": cluster_features_list}

    if request.method == "POST":
        # load the JSON Data
        run_data = json.loads(request.body)
        model_params = run_data.get("model")
        dataset_params = run_data.get("dataset")

        tickerString = dataset_params.get("tickers")
        tickers = tickerString.split(",")
        tickers = [ticker.strip().upper() for ticker in tickers]

        start_date = dataset_params.get("start_date")
        end_date = dataset_params.get("end_date")

        steps = dataset_params.get("steps")
        interval = dataset_params.get("interval")

        cluster_features = dataset_params.get("cluster_features")
        training_features_cats = dataset_params.get("training_features")

        feature_sample_size = dataset_params.get("feature_sample_size")
        feature_sample_num = dataset_params.get("feature_sample_num")

        

        # target_features = ['sumpctChgclose_1','sumpctChgclose_2','sumpctChgclose_3','sumpctChgclose_4','sumpctChgclose_5','sumpctChgclose_6']
        target_features = ['pctChgclose-14','pctChgclose-13','pctChgclose-12','pctChgclose-11','pctChgclose-10','pctChgclose-9','pctChgclose-8','pctChgclose-7','pctChgclose-6','pctChgclose-5','pctChgclose-4','pctChgclose-3','pctChgclose-2','pctChgclose-1','pctChgclose-0','pctChgclose+1','pctChgclose+2','pctChgclose+3','pctChgclose+4','pctChgclose+5','pctChgclose+6']

        scaling_dict = {
            "price_vars": SP.ScalingMethod.UNSCALED,
            "trend_vars": SP.ScalingMethod.UNSCALED,
            "pctChg_vars": SP.ScalingMethod.QUANT_MINMAX,
            "rolling_vars": SP.ScalingMethod.QUANT_MINMAX,
            "target_vars": SP.ScalingMethod.QUANT_MINMAX,
        }

        col_dict = {
            "PctChgVars": supported_params.pct_chg_features,
            "CumulativeVars": supported_params.cuma_features,
            "RollingVars": supported_params.rolling_features,
            "PriceVars": supported_params.price_features,
            "TrendVars": supported_params.trend_features,
        }

        training_features = []
        for feature in training_features_cats:
            training_features += col_dict[feature]

        # Process the data (this is where you would include your logic)
        group_params = CP.StockClusterGroupParams.objects.create(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            n_steps=steps,
            cluster_features=cluster_features,
            target_cols=target_features,
            interval=interval,
            training_features=training_features,
        )
        group_params.initialize()
        group_params.set_scaling_dict(scaling_dict)
        group_params.feature_sample_size = feature_sample_size
        group_params.feature_sample_num = feature_sample_num
        group_params.model_params = model_params
        group_params.initialize()
        group_params.create_model_dir()
        group_params.save()

        CreateGroupBackground(group_params).start()

        return HttpResponse("Run Started Sucessfully")

    return render(request, "ClusterPipeline/create_run.html", context)


@csrf_exempt
def cluster_group(request):
    if request.method == "GET":
        cluster_group_list = CP.StockClusterGroup.objects.all()
        item_list = [(item.pk, item.group_params.name) for item in cluster_group_list]
        return render(
            request, "ClusterPipeline/cluster_group.html", {"menu_items": item_list}
        )
    elif request.method == "POST":
        cluster_group_value = request.POST.get("cluster_group")
        group_data = ast.literal_eval(cluster_group_value)
        id = group_data[0]
        print(id)
        return redirect("cluster_group_detail", id=id)


@csrf_exempt
def cluster_group_detail(request, id):
    cluster_group = get_cluster_group(id)

    cluster_results = []

    for cluster in cluster_group.clusters:
        if len(cluster.models) == 0:
            continue
        fig1 = cluster.visualize_cluster_2d()
        sorted_models = cluster.sort_models()
        fig2 = sorted_models[0].visualize_future_distribution()
        fig1_json = json.loads(plotly.io.to_json(fig1))
        fig2_json = json.loads(plotly.io.to_json(fig2))

        best_model = sorted_models[0]
        metrics = best_model.generate_results()
        model_config = best_model.generate_model_config()

        cluster_dict = {}
        cluster_dict["figs"] = (fig1_json, fig2_json)
        cluster_dict["metrics"] = metrics
        cluster_dict["model_config"] = model_config
        cluster_dict["model_stats"] = best_model.model_metrics

        cluster_results.append(cluster_dict)

    cluster_results = json.dumps(cluster_results)

    item_list = [
        (item.pk, item.group_params.name) for item in CP.StockClusterGroup.objects.all()
    ]

    return render(
        request,
        "ClusterPipeline/cluster_group_detail.html",
        {"results": cluster_results, "menu_items": item_list},
    )


@csrf_exempt
def cluster_detail(request, group_id, cluster_id):
    cluster_group = get_cluster_group(group_id)
    cluster = cluster_group.clusters[cluster_id]

    model_results = []

    sorted_models = cluster.sort_models()
    if len(sorted_models) == 0:
        return render(
            request, "ClusterPipeline/cluster_detail.html", {"results": model_results}
        )

    for model in sorted_models:
        fig1 = model.visualize_future_distribution()
        fig1_json = json.loads(plotly.io.to_json(fig1))

        fig2 = model.visualize_future_distribution(isTest=False)
        fig2_json = json.loads(plotly.io.to_json(fig2))

        metrics = model.generate_results()
        model_config = model.generate_model_config()

        model_dict = {}
        model_dict["figs"] = [fig1_json, fig2_json]
        model_dict["metrics"] = metrics
        model_dict["model_config"] = model_config
        model_dict["model_stats"] = model.model_metrics

        model_results.append(model_dict)

    model_results = json.dumps(model_results)

    return render(
        request, "ClusterPipeline/cluster_detail.html", {"results": model_results}
    )


@csrf_exempt
def prediction_detail(request, prediction_id):
    """
    View/Controller for the prediction detail page. The processing for the predictions has been
    completed at this point so we are simply retreiving and passing along the dataframe here
    """

    if request.method == "POST":
        data = json.loads(request.body)
        status_arr = data.get("status_arr")

        prediction = Pred.StockPrediction.objects.get(pk=prediction_id)
        prediction.initialize()
        df = prediction.load_data_frame()
        df["status"] = status_arr
        prediction.save_data_frame(df)
        prediction.save()

        return JsonResponse({"success": True})
    
    groups = CP.StockClusterGroup.objects.all()
    tickers = []
    for group in groups:
        tickers += group.group_params.tickers
    tickers = list(set(tickers))

    start_date = "2020-01-01"
    prediction = Pred.StockPrediction.objects.get(pk=prediction_id)
    prediction.create_general_data_set(
        start_date=start_date, end_date=date.today().strftime("%Y-%m-%d")
    )
    prediction_df = prediction.load_data_frame()
    pred_df_json = prediction_df.to_json(orient="split", date_format="iso")

    close_df = yf.download(
        prediction.ticker,
        start=start_date,
        end=(date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=prediction.interval,
    )
    close_df = close_df[["Close"]].reset_index()
    close_df = close_df.rename(columns={"Close": "close"})

    close_df_json = close_df.to_json(orient="split", date_format="iso")

    return render(
        request,
        "ClusterPipeline/forcast_detail.html",
        {
            "stock_predictions": json.dumps([{'prediction_id': prediction.pk,  'start_date' : prediction.prediction_start_date.strftime("%Y-%m-%d")}]),
            "pred_df": pred_df_json,
            "close_df": close_df_json,
            "tickers": tickers,
            "prediction_id": prediction_id,
            "ticker": prediction.ticker,
            "interval": prediction.interval,
            "prediction_start_date": prediction.prediction_start_date.strftime(
                "%Y-%m-%d"
            ),
        },
    )


@csrf_exempt
def forcast(request):
    """
    View/Controller for the prediction page. This is where prediction is loaded/created.
    If the prediction already exists, we simply redirect to the prediction detail page.
    if the prediction does not exist, we create the prediction and redirect to the prediction detail page
    """
    groups = CP.StockClusterGroup.objects.all()
    tickers = []
    for group in groups:
        tickers += group.group_params.tickers

    tickers = list(set(tickers))

    if request.GET:
        ticker = request.GET.get("ticker")
        interval = request.GET.get("interval")
        prediction_start_date = request.GET.get("start")
        prediction_end_date = request.GET.get("end")

        if Pred.StockForcastTimeline.objects.filter(
            ticker=ticker,
            interval=interval,
        ).exists():
            print("prediction exists")
            forcast_timeline = Pred.StockForcastTimeline.objects.get(
                ticker=ticker,
                interval=interval,
            )
            forcast_timeline.initialize()

            if (
                datetime.strptime(prediction_start_date, "%Y-%m-%d").date()
                < forcast_timeline.prediction_start_date
                or datetime.strptime(prediction_end_date, "%Y-%m-%d").date()
                > forcast_timeline.prediction_end_date
            ):
                forcast_timeline.add_prediction_range(
                    start_date=prediction_start_date,
                    end_date=prediction_end_date,
                    total_model_accuracy_thresh=60,
                    individual_model_accuracy_thresh=64,
                    epochs_threshold=6,
                )
            forcast_timeline.save()
            forcast_id = forcast_timeline.pk

            return redirect("forcast_detail", forcast_id=forcast_id)
        else:
            print("prediction does not exist")

            forcast_timeline = Pred.StockForcastTimeline(
                ticker=ticker,
                interval=interval,
                prediction_start_date=prediction_start_date,
                prediction_end_date=prediction_end_date,
            )
            forcast_timeline.save()
            forcast_timeline.initialize()
            forcast_timeline.add_prediction_range(
                start_date=prediction_start_date,
                end_date=prediction_end_date,
                total_model_accuracy_thresh=60,
                individual_model_accuracy_thresh=64,
                epochs_threshold=6,
            )
            forcast_timeline.save()
            forcast_id = forcast_timeline.pk

        return redirect("forcast_detail", forcast_id=forcast_id)

    return render(request, "ClusterPipeline/predictions.html", {"tickers": tickers})


@csrf_exempt
def forcast_detail(request, forcast_id):
    """
    View/Controller for the prediction detail page. The processing for the predictions has been
    completed at this point so we are simply retreiving and passing along the dataframe here
    """

    if request.method == "POST":
        data = json.loads(request.body)
        status_arr = data.get("status_arr")

        forcast = Pred.StockForcastTimeline.objects.get(pk=forcast_id)
        forcast.initialize()
        forcast.load_data_frame()
        df = forcast.forcast_dataframe
        df["status"] = status_arr
        for prediction in forcast.stock_predictions: 
            new_df = df[df['start_date'] == prediction.prediction_start_date.strftime("%Y-%m-%d")]
            # drop the columns where the whole columns is null
            new_df = new_df.dropna(axis=1, how='all')
            prediction.save_data_frame(new_df)
            prediction.save()


        forcast.forcast_dataframe = df
        forcast.save_data_frame(df)
        forcast.save()

        return JsonResponse({"success": True})
    
    groups = CP.StockClusterGroup.objects.all()
    tickers = []
    for group in groups:
        tickers += group.group_params.tickers
    tickers = list(set(tickers))

    start_date = "2020-01-01"
    forcast = Pred.StockForcastTimeline.objects.get(pk=forcast_id)
    forcast.initialize()
    prediction = forcast.stock_predictions[0]

    prediction.create_general_data_set(
        start_date=start_date, end_date=date.today().strftime("%Y-%m-%d")
    )

    forcast.rebuild_data_frame()
    forcast_df = forcast.load_data_frame()
    forcast_df_json = forcast_df.to_json(orient="split", date_format="iso")

    close_df = yf.download(
        prediction.ticker,
        start=start_date,
        end=(date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=prediction.interval,
    )

    close_df = close_df[["Close"]].reset_index()
    close_df = close_df.rename(columns={"Close": "close"})

    close_df_json = close_df.to_json(orient="split", date_format="iso")

    stock_predictions = []
    for prediction in forcast.stock_predictions: 
        stock_predictions.append({'prediction_id': prediction.pk,  'start_date' : prediction.prediction_start_date.strftime("%Y-%m-%d")})

    return render(
        request,
        "ClusterPipeline/forcast_detail.html",
        {
            "stock_predictions": json.dumps(stock_predictions),
            "pred_df": forcast_df_json,
            "close_df": close_df_json,
            "tickers": tickers,
            "prediction_id": forcast_id,
            "ticker": prediction.ticker,
            "interval": prediction.interval,
            "prediction_start_date": prediction.prediction_start_date.strftime(
                "%Y-%m-%d"
            ),
        },
    )

def get_prediction_vis_files(request, prediction_id):
    try: 
        prediction = Pred.StockPrediction.objects.get(pk=prediction_id)
    except Pred.StockPrediction.DoesNotExist:
        return JsonResponse({'error': 'Prediction not found'}, status=404)
    
    prediction_dir = os.path.join(prediction.dir_path, 'clusters')

    if not os.path.exists(prediction_dir):
        return JsonResponse({'error': 'Prediction not found'}, status=404)
    
    files_data = {}
    for filename in os.listdir(prediction_dir):
        if filename.endswith(".html"):
            cluster_id = filename.replace('cluster-', '').replace('.html', '')
            with open(os.path.join(prediction_dir, filename)) as f:
                files_data[cluster_id] = f.read()
    
    return JsonResponse(files_data)

def get_models(request, model_ids):
    model_ids = model_ids.split(',')
    models = RNN.RNNModel.objects.filter(pk__in=model_ids)
    model_results = []
    for model in models:
        fig = model.visualize_future_distribution()
        fig_json = json.loads(plotly.io.to_json(fig))
        metrics = model.generate_results()
        model_config = model.generate_model_config()

        model_dict = {}
        model_dict["figs"] = [fig_json]
        model_dict["metrics"] = metrics
        model_dict["model_config"] = ""
        model_dict["model_stats"] = model.model_metrics

        model_results.append(model_dict)

    return JsonResponse(model_results, safe=False)





def get_cluster_group(id):
    cluster_group = cache.get(f"cluster_group_{id}")

    if not cluster_group:
        cluster_group = CP.StockClusterGroup.objects.get(pk=id)
        cluster_group.load_saved_group()
        cache.set(f"cluster_group_{id}", cluster_group, 60 * 60 * 15)

    return cluster_group
