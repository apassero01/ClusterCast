# Generated by Django 4.1 on 2023-11-15 04:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ClusterPipeline", "0010_remove_sequenceelement_seq_x_scaled_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="sequenceelement",
            name="seq_x_scaled_string",
            field=models.TextField(default=""),
        ),
        migrations.AddField(
            model_name="sequenceelement",
            name="seq_y_scaled_string",
            field=models.TextField(default=""),
        ),
    ]
