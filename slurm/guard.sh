# Slurm account guard. Source it from the site env file, which every job in slurm/ sources first.
# The site env sets the policy; this file holds no site names:
#   FLASHEMBED_ALLOWED_ACCOUNTS    extended regex that an account and a QOS must match, e.g. '^(acct1|acct2)(-b)?$'
#   FLASHEMBED_FORBIDDEN_ACCOUNTS  extended regex that must never match (checked first)
# With neither set, nothing is restricted.
# Inside a job: if the job's account or QOS fails the policy, it prints why, cancels the job and exits 1, so nothing
# runs or writes under the wrong allocation. Outside a job it defines `flashembed_sbatch`, a submit wrapper that
# refuses -A/--account/-q/--qos values (and SBATCH_ACCOUNT/SBATCH_QOS) failing the policy, submitting nothing; after a
# submit it re-checks the account and QOS Slurm actually assigned and cancels the job if either fails.

_fe_acct_ok() {  # $1 = account or QOS
  local v=$1
  [[ -z "${FLASHEMBED_ALLOWED_ACCOUNTS:-}${FLASHEMBED_FORBIDDEN_ACCOUNTS:-}" ]] && return 0
  [[ -z "$v" ]] && return 1
  [[ -n "${FLASHEMBED_FORBIDDEN_ACCOUNTS:-}" && "$v" =~ $FLASHEMBED_FORBIDDEN_ACCOUNTS ]] && return 1
  [[ -z "${FLASHEMBED_ALLOWED_ACCOUNTS:-}" || "$v" =~ $FLASHEMBED_ALLOWED_ACCOUNTS ]]
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _fj=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)
  _fa=${SLURM_JOB_ACCOUNT:-$(sed -n 's/.* Account=\([^ ]*\).*/\1/p' <<< "$_fj" | head -1)}
  _fq=${SLURM_JOB_QOS:-$(sed -n 's/.* QOS=\([^ ]*\).*/\1/p' <<< "$_fj" | head -1)}
  if ! _fe_acct_ok "$_fa" || ! _fe_acct_ok "$_fq"; then
    echo "BLOCKED by slurm/guard.sh: job $SLURM_JOB_ID has account='$_fa' qos='$_fq', which the site policy forbids." >&2
    scancel "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}" 2>/dev/null
    exit 1
  fi
  unset _fj _fa _fq
fi

flashembed_sbatch() {
  local a=${SBATCH_ACCOUNT:-} q=${SBATCH_QOS:-} prev="" x
  for x in "$@"; do
    case "$prev" in -A|--account) a=$x ;; -q|--qos) q=$x ;; esac
    case "$x" in
      --account=*) a=${x#*=} ;; --qos=*) q=${x#*=} ;;
      -A?*) a=${x#-A} ;; -q?*) q=${x#-q} ;;
    esac
    prev=$x
  done
  if ! _fe_acct_ok "$a" || ! _fe_acct_ok "$q"; then
    echo "REFUSED by slurm/guard.sh: account='$a' qos='$q' not allowed by the site policy; nothing submitted." >&2
    return 1
  fi
  local out id real v
  out=$(sbatch --parsable "$@") || return 1
  id=${out%%;*}
  real=$(squeue -h -j "$id" -o "%a %q" 2>/dev/null | head -1)
  for v in $real; do
    if ! _fe_acct_ok "$v"; then
      scancel "$id"; echo "CANCELLED $id: Slurm assigned '$real'" >&2; return 1
    fi
  done
  echo "$id"
}
