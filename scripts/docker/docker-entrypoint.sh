#!/bin/bash
set -euo pipefail


echo "INFO: Running '${FR_SLUG}' docker-entrypoint.sh..."

_doStart()
{
	sleep 2
	exec python3 examples/clients/main.py || exit 2
	# exec python -u ./main.py || exit 2
	exit 0
}


main()
{
	umask 0002 || exit 2
	find "${FR_HOME_DIR}" "${FR_DATA_DIR}" "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -path "*/modules" -prune -o -name ".env" -o -print0 | sudo xargs -0 chown -c "${USER}:${GROUP}" || exit 2
	find "${FR_DIR}" "${FR_DATA_DIR}" -type d -not -path "*/modules/*" -not -path "*/scripts/*" -exec sudo chmod 770 {} + || exit 2
	find "${FR_DIR}" "${FR_DATA_DIR}" -type f -not -path "*/modules/*" -not -path "*/scripts/*" -exec sudo chmod 660 {} + || exit 2
	find "${FR_DIR}" "${FR_DATA_DIR}" -type d -not -path "*/modules/*" -not -path "*/scripts/*" -exec sudo chmod ug+s {} + || exit 2
	find "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -type d -exec sudo chmod 775 {} + || exit 2
	find "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -type f -exec sudo chmod 664 {} + || exit 2
	find "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -type d -exec sudo chmod +s {} + || exit 2
	# echo "${USER} ALL=(ALL) ALL" | sudo tee -a "/etc/sudoers.d/${USER}" > /dev/null || exit 2
	echo ""

	## Parsing input:
	case ${1:-} in
		"" | -s | --start | start | --run | run)
			_doStart;;
			# shift;;
		-b | --bash | bash | /bin/bash)
			shift
			if [ -z "${*:-}" ]; then
				echo "INFO: Starting bash..."
				/bin/bash
			else
				echo "INFO: Executing command -> ${*}"
				exec /bin/bash -c "${@}" || exit 2
			fi
			exit 0;;
		*)
			echo "ERROR: Failed to parsing input -> ${*}"
			echo "USAGE: ${0}  -s, --start, start | -b, --bash, bash, /bin/bash"
			exit 1;;
	esac
}

main "${@:-}"
