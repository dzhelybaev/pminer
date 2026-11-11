#!/usr/bin/env bash
# Аргументы из Flight Sheet: пул (первый из CUSTOM_URL), кошелёк.воркер, пароль, доп. параметры
url=$(echo "$CUSTOM_URL" | awk '{print $1}')
args=""
[[ -n $url ]] && args="--pool $url --wallet $CUSTOM_TEMPLATE --pass ${CUSTOM_PASS:-x}"
echo "$args $CUSTOM_USER_CONFIG" > "$CUSTOM_CONFIG_FILENAME"
