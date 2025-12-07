#!/bin/bash
# ============================================================================
# Athena Kafka Topics Setup
# Creates all required topics with appropriate configurations
# ============================================================================

set -e  # Exit on error

echo "🚀 Creating Athena Kafka Topics..."

# Kafka broker
KAFKA_BROKER="localhost:29092"

# Topic configurations
# Format: "topic_name:partitions:replication_factor:retention_ms"
TOPICS=(
    # Data Ingestion Topics
    "raw_commits:6:1:604800000"           # 7 days retention
    "raw_issues:3:1:604800000"            # 7 days retention
    "raw_prs:3:1:604800000"               # 7 days retention
    "raw_files:6:1:604800000"             # 7 days retention

    # Feature Engineering Topics
    "parsed_ast:6:1:259200000"            # 3 days retention
    "extracted_features:6:1:259200000"    # 3 days retention
    "code_embeddings:6:1:259200000"       # 3 days retention

    # ML Pipeline Topics
    "training_data:3:1:86400000"          # 1 day retention
    "predictions:6:1:2592000000"          # 30 days retention
    "model_updates:1:1:31536000000"       # 1 year retention

    # RL Topics
    "feedback_events:3:1:2592000000"      # 30 days retention
    "rl_trajectories:3:1:604800000"       # 7 days retention
    "policy_updates:1:1:31536000000"      # 1 year retention

    # Pattern Mining Topics
    "pattern_discoveries:3:1:2592000000"  # 30 days retention
    "association_rules:3:1:2592000000"    # 30 days retention

    # System Topics
    "crawler_events:3:1:86400000"         # 1 day retention
    "errors:3:1:604800000"                # 7 days retention
    "metrics:3:1:86400000"                # 1 day retention

    # Dead Letter Queues
    "dlq_commits:3:1:2592000000"          # 30 days retention
    "dlq_features:3:1:2592000000"         # 30 days retention
    "dlq_predictions:3:1:2592000000"      # 30 days retention
)

# Create topics
for topic_config in "${TOPICS[@]}"; do
    IFS=':' read -r topic partitions replication retention <<< "$topic_config"

    echo "📝 Creating topic: $topic (partitions=$partitions, retention=${retention}ms)"

    docker exec athena-kafka kafka-topics \
        --create \
        --if-not-exists \
        --bootstrap-server localhost:9092 \
        --topic "$topic" \
        --partitions "$partitions" \
        --replication-factor "$replication" \
        --config retention.ms="$retention" \
        --config compression.type=snappy \
        --config cleanup.policy=delete
done

echo ""
echo "✅ All topics created successfully!"
echo ""
echo "📊 Topic Summary:"
docker exec athena-kafka kafka-topics \
    --list \
    --bootstrap-server localhost:9092 | grep -E "^(raw_|parsed_|extracted_|training_|predictions|feedback_|rl_|pattern_|association_|crawler_|errors|metrics|dlq_)"

echo ""
echo "🔍 Topic Details (sample):"
docker exec athena-kafka kafka-topics \
    --describe \
    --bootstrap-server localhost:9092 \
    --topic raw_commits

echo ""
echo "✨ Kafka is ready for Athena!"
echo ""
echo "📌 Usage Examples:"
echo ""
echo "# Check topic"
echo "docker exec athena-kafka kafka-topics --describe --bootstrap-server localhost:9092 --topic raw_commits"
echo ""
echo "# Monitor messages"
echo "docker exec athena-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic raw_commits --from-beginning --max-messages 10"
echo ""
echo "# Check consumer groups"
echo "docker exec athena-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --list"
echo ""
