# Generated by Django 4.1 on 2024-02-15 23:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ClusterPipeline", "0034_supportedparams_lag_features"),
    ]

    operations = [
        migrations.AddField(
            model_name="supportedparams",
            name="momentum_features",
            field=models.JSONField(default=list),
        ),
    ]
