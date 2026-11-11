#!/usr/bin/env bash
# Вызывается агентом HiveOS (source); должен выставить $khs и $stats
F=/run/pminer/stats.json
. /hive/miners/custom/pminer/h-manifest.conf
if [[ ! -s $F || $(( $(date +%s) - $(stat -c %Y "$F") )) -gt 30 ]]; then
  khs=0
  stats=null
else
  busid=$(jq -c '.busid' "$F")
  pick='. as $g | [$b[] as $x | (($g.busids // []) | map(ascii_downcase) | index($x)) as $i
        | if $i == null then 0 else $g[$k][$i] end]'
  temp=$(jq -c --argjson b "$busid" --arg k temp "$pick" <<< "$gpu_stats" 2>/dev/null) || temp='[]'
  fan=$(jq -c --argjson b "$busid" --arg k fan "$pick" <<< "$gpu_stats" 2>/dev/null) || fan='[]'
  [[ -z $temp ]] && temp='[]'; [[ -z $fan ]] && fan='[]'
  khs=$(jq '[.hs[]] | add / 1000' "$F")
  stats=$(jq -c --argjson t "$temp" --argjson f "$fan" \
    '{hs:[.hs[] | ./1e6 | floor], hs_units:"mhs", temp:$t, fan:$f, uptime:.uptime, ver:.ver,
      ar:[0,0], algo:"pearlhash", bus_numbers:.bus}' "$F")
fi
[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats=null
