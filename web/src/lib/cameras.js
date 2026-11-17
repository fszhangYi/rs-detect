export function pickCandidateSerial(candidates, preferred) {
  const list = candidates || [];
  if (!list.length) return undefined;
  const hasReal = list.some((d) => !d.is_mock);
  if (preferred && list.some((d) => d.serial === preferred)) {
    const pref = list.find((d) => d.serial === preferred);
    if (pref?.is_mock && hasReal) return list.find((d) => !d.is_mock).serial;
    return preferred;
  }
  const real = list.find((d) => !d.is_mock);
  return (real || list[0]).serial;
}

