# Generated by Django 4.1 on 2024-01-08 00:32

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("ClusterPipeline", "0025_alter_stockclustergroupparams_name"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="stockclustergroupparams",
            name="X_feature_dict",
        ),
        migrations.RemoveField(
            model_name="stockclustergroupparams",
            name="y_feature_dict",
        ),
    ]
