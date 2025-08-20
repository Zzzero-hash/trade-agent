#!/bin/sh
if [ -z "$husky_skip_init" ]; then
  debug () {
    [ "$HUSKY_DEBUG" = "1" ] && echo "husky (debug) - $1"
  }
  readonly husky_skip_init=1
  export husky_skip_init
  debug "starting..."
  if [ "$HUSKY" = "0" ]; then
    debug "HUSKY env variable is set to 0, skipping hook"
    exit 0
  fi
  if [ ! -f package.json ]; then
    debug "package.json not found, skipping hooks"
    exit 0
  fi
  command -v npm >/dev/null 2>&1 || { echo >&2 "husky - npm not found, skipping hooks"; exit 0; }
  hook_name="$(basename "$0")"
  if [ "$HUSKY_DEBUG" = "1" ]; then
    echo "husky (debug) - hook: $hook_name"
  fi
  export readonly husky_skip_init
  sh -e "$0" "$@"
  exitCode="$?"
  if [ $exitCode != 0 ]; then
    echo "husky - $hook_name hook exited with code $exitCode (error)"
    exit $exitCode
  fi
  exit 0
fi
