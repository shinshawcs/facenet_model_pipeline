from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from kubernetes.client import models as k8s
from datetime import timedelta

def check_latency(**kwargs):
    with open("/opt/airflow/shared/models/latency.txt", "r") as f:
        latency = float(f.read().strip())
        print(f"✅ Captured latency: {latency}")
        if latency > 2.0:
            raise ValueError(f"❌ Latency too high: {latency} ms")

print("✅ current version DAG facenet_pipeline loaded")
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='facenet_pipeline',
    default_args=default_args,
    schedule_interval=None,  
    start_date=days_ago(1),
    catchup=False,
    description='FaceNet full pipeline: fine_tune -> compile ',
) as dag:
    model_finetune = KubernetesPodOperator(
        task_id="facenet_finetune_gpu",
        name="facenet_finetune",
        namespace="airflow",
        image="shinshaw/facenet-finetune:latest",
        cmds=["python", "/app/facenet_finetune.py"],
        image_pull_policy="Always",
        container_resources=k8s.V1ResourceRequirements(
            limits={"nvidia.com/gpu": "1"}
        ),
        tolerations=[
            k8s.V1Toleration(
                key="nvidia.com/gpu",
                operator="Exists",
                effect="NoSchedule"
            )
        ],
        node_selector={"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"},
        env_vars={
            "PVC_DATA_PATH": "/opt/airflow/shared/test_images_cropped",
            "GCS_BUCKET": "my-facenet-bucket", 
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/credentials.json"
        },
        volumes=[
            k8s.V1Volume(
                name="airflow-data",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name="airflow-data-pvc"
                )
            ),
            k8s.V1Volume(
                name="gcs-credentials",
                secret=k8s.V1SecretVolumeSource(
                    secret_name="gcp-storage-credentials"
                )
            )
        ],
        volume_mounts=[
            k8s.V1VolumeMount(
                name="airflow-data",
                mount_path="/opt/airflow/shared"
            ),
            k8s.V1VolumeMount(
                name="gcs-credentials",
                mount_path="/app/credentials.json",
                sub_path="credentials.json",
                read_only=True
            )
        ],
        is_delete_operator_pod=True,
        get_logs=True
    )
    model_compile = KubernetesPodOperator(
        task_id="model_compile_gpu",
        name="facenet_compile",
        namespace="airflow",
        image="shinshaw/facenet-compile:latest",
        cmds=["python", "/app/facenet_compile.py"],
        image_pull_policy="Always",
        container_resources=k8s.V1ResourceRequirements(
            limits={"nvidia.com/gpu": "1"}
        ),
        tolerations=[
            k8s.V1Toleration(
                key="nvidia.com/gpu",
                operator="Exists",
                effect="NoSchedule"
            )
        ],
        node_selector={"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"},
        env_vars={
            "MODEL_BASE_PATH": "/opt/airflow/shared/models",
            "GCS_BUCKET": "my-facenet-bucket",
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/credentials.json"
        },
        volumes=[
            k8s.V1Volume(
                name="airflow-data",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name="airflow-data-pvc"
                )
            ),
            k8s.V1Volume(
                name="gcs-credentials",
                secret=k8s.V1SecretVolumeSource(
                    secret_name="gcp-storage-credentials"
                )
            )
        ],
        volume_mounts=[
            k8s.V1VolumeMount(
                name="airflow-data",
                mount_path="/opt/airflow/shared"
            ),
            k8s.V1VolumeMount(
                name="gcs-credentials",
                mount_path="/app/credentials.json",
                sub_path="credentials.json",
                read_only=True
            )
        ],
        is_delete_operator_pod=True,
        get_logs=True
    )
    model_benchmark = KubernetesPodOperator(
        task_id="model_benchmark_gpu",
        name="facenet_benchmark",
        namespace="airflow",
        image="shinshaw/facenet-benchmark:latest",
        cmds=["python", "/app/benchmark.py"],
        image_pull_policy="Always",
        container_resources=k8s.V1ResourceRequirements(
            limits={"nvidia.com/gpu": "1"}
        ),
        tolerations=[
            k8s.V1Toleration(
                key="nvidia.com/gpu",
                operator="Exists",
                effect="NoSchedule"
            )
        ],
        node_selector={"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"},
        env_vars={
            "MODEL_BASE_PATH": "/opt/airflow/shared/models",
            "GCS_BUCKET": "my-facenet-bucket",
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/credentials.json"
        },
        volumes=[
            k8s.V1Volume(
                name="airflow-data",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name="airflow-data-pvc"
                )
            ),
            k8s.V1Volume(
                name="gcs-credentials",
                secret=k8s.V1SecretVolumeSource(
                    secret_name="gcp-storage-credentials"
                )
            )
        ],
        volume_mounts=[
            k8s.V1VolumeMount(
                name="airflow-data",
                mount_path="/opt/airflow/shared"
            ),
            k8s.V1VolumeMount(
                name="gcs-credentials",
                mount_path="/app/credentials.json",
                sub_path="credentials.json",
                read_only=True
            )
        ],
        is_delete_operator_pod=True,
        get_logs=True
    )
    model_evaluate = KubernetesPodOperator(
        task_id="model_evaluate_gpu",
        name="facenet_evaluate",
        namespace="airflow",
        image="shinshaw/facenet-evaluate:latest",
        cmds=["python", "/app/facenet_evaluate.py"],
        image_pull_policy="Always",
        container_resources=k8s.V1ResourceRequirements(
            limits={"nvidia.com/gpu": "1"}
        ),
        tolerations=[
            k8s.V1Toleration(
                key="nvidia.com/gpu",
                operator="Exists",
                effect="NoSchedule"
            )
        ],
        node_selector={"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"},
        env_vars={
            "MODEL_BASE_PATH": "/opt/airflow/shared/models",
            "PVC_DATA_PATH": "/opt/airflow/shared/test_images_cropped",
            "GCS_BUCKET": "my-facenet-bucket",
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/credentials.json"
        },
        volumes=[
            k8s.V1Volume(
                name="airflow-data",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name="airflow-data-pvc"
                )
            ),
            k8s.V1Volume(
                name="gcs-credentials",
                secret=k8s.V1SecretVolumeSource(
                    secret_name="gcp-storage-credentials"
                )
            )
        ],
        volume_mounts=[
            k8s.V1VolumeMount(
                name="airflow-data",
                mount_path="/opt/airflow/shared"
            ),
            k8s.V1VolumeMount(
                name="gcs-credentials",
                mount_path="/app/credentials.json",
                sub_path="credentials.json",
                read_only=True
            )
        ],
        is_delete_operator_pod=True,
        get_logs=True
    )
    check_latency_task = PythonOperator(
        task_id="check_latency",
        python_callable=check_latency
    )
    notify_latency_alert = EmailOperator(
        task_id="notify_latency_alert",
        to="egatech2017@email.com",
        subject="🚨 Latency Alert",
        html_content="Latency from benchmark exceeded threshold.",
        trigger_rule="one_failed"  
    )
    model_deploy = KubernetesPodOperator(
        task_id="model_deploy_gpu",
        name="facenet_deploy",
        namespace="airflow",
        image="shinshaw/facenet-deploy:latest",  
        cmds=["python", "/app/facenet_deploy.py"],  
        image_pull_policy="Always",
        container_resources=k8s.V1ResourceRequirements(
            limits={"nvidia.com/gpu": "1"}
        ),
        tolerations=[
            k8s.V1Toleration(
                key="nvidia.com/gpu",
                operator="Exists",
                effect="NoSchedule"
            )
        ],
        node_selector={"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"},
        env_vars={
            "MODEL_BASE_PATH": "/opt/airflow/shared/models",
            "GCS_BUCKET": "my-facenet-bucket",
            "GOOGLE_APPLICATION_CREDENTIALS": "/app/credentials.json"
        },
        volumes=[
            k8s.V1Volume(
                name="airflow-data",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name="airflow-data-pvc"
                )
            ),
            k8s.V1Volume(
                name="gcs-credentials",
                secret=k8s.V1SecretVolumeSource(
                    secret_name="gcp-storage-credentials"
                )
            )
        ],
        volume_mounts=[
            k8s.V1VolumeMount(
                name="airflow-data",
                mount_path="/opt/airflow/shared"
            ),
            k8s.V1VolumeMount(
                name="gcs-credentials",
                mount_path="/app/credentials.json",
                sub_path="credentials.json",
                read_only=True
            )
        ],
        is_delete_operator_pod=True,
        get_logs=True
        )
    #model_finetune model_compile model_evaluate model_benchmark check_latency_task notify_latency_alert model_deploy
    model_finetune >> model_compile >> [model_benchmark, model_evaluate] 
    model_benchmark >> check_latency_task
    check_latency_task >> notify_latency_alert
    check_latency_task >> model_deploy
