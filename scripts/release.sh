#!/bin/bash
set -euo pipefail


## --- Base --- ##
# Getting path of this script file:
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
_PROJECT_DIR="$(cd "${_SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${_PROJECT_DIR}" || exit 2

# Loading base script:
# shellcheck disable=SC1091
source ./scripts/base.sh

# Loading .env file (if exists):
if [ -f ".env" ]; then
	# shellcheck disable=SC1091
	source .env
fi

if [ -z "$(which gh)" ]; then
	echoError "'gh' not found or not installed."
	exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
	echoError "You need to login: gh auth login"
	exit 1
fi
## --- Base --- ##


## --- Main --- ##
main()
{
	_current_version="$(./scripts/get-version.sh)"
	echoInfo "Creating release for version: 'v${_current_version}'..."
	gh release create "v${_current_version}" --generate-notes
	echoOk "Done."
}

main
## --- Main --- ##
