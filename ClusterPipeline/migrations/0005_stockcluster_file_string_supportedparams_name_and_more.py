# Generated by Django 4.1 on 2023-11-13 22:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("ClusterPipeline", "0004_stepresult"),
    ]

    operations = [
        migrations.AddField(
            model_name="stockcluster",
            name="file_string",
            field=models.CharField(default="", max_length=100),
        ),
        migrations.AddField(
            model_name="supportedparams",
            name="name",
            field=models.CharField(default="", max_length=100),
        ),
        migrations.AlterField(
            model_name="stockcluster",
            name="cluster_group",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="clusters_obj",
                to="ClusterPipeline.stockclustergroup",
            ),
        ),
    ]
