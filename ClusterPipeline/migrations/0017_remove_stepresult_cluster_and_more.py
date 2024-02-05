# Generated by Django 4.1 on 2023-12-16 23:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        (
            "ClusterPipeline",
            "0016_stepresult_actual_values_stepresult_predicted_values",
        ),
    ]

    operations = [
        migrations.RemoveField(
            model_name="stepresult",
            name="cluster",
        ),
        migrations.RemoveField(
            model_name="stockcluster",
            name="elements_file_string",
        ),
        migrations.RemoveField(
            model_name="stockcluster",
            name="model_file_string",
        ),
        migrations.CreateModel(
            name="RNNModel",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("model_features", models.CharField(default="None", max_length=1000)),
                ("summary_string", models.CharField(default="None", max_length=1000)),
                ("model_dir", models.CharField(default="None", max_length=1000)),
                (
                    "cluster",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="RNNModels",
                        to="ClusterPipeline.stockcluster",
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="stepresult",
            name="RNNModel",
            field=models.ForeignKey(
                default=1,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="model_results",
                to="ClusterPipeline.rnnmodel",
            ),
            preserve_default=False,
        ),
    ]