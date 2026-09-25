const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  voices: [], selected: null, filter: 'all', mode: 'catalog',
  creation: 'voice', musicKind: 'song', musicPresets: [], musicPreset: null,
  musicPresetQuery: '', musicCategory: 'all',
  sfxKind: 'oneshot', sfxPresets: [], sfxPreset: null,
  sfxPresetQuery: '', sfxCategory: 'all', latestPromptKind: null,
  generating: false, startedAt: 0, timer: null, queueTimer: null,
  progress: 4, history: [], latestPromptJob: null,
};

const prompts = {
  warm: "Hey. You made it. I was hoping I'd get a minute alone with you.",
  dramatic: "Close the door and sit down. Before you start apologizing, take a breath. I don't need a polished story; I need the version that actually happened.",
  playful: "There you are! I wasn't worried. I just happened to check the window, the hallway, and my phone. Come here and make your explanation adorable.",
  long: "The rain had finally stopped by the time the last train arrived. I watched the empty platform reflected in the glass and wondered whether you would recognize me after all this time. Then the doors opened, you stepped out, and every careful sentence I had prepared disappeared. So I smiled and said the only honest thing left: I'm glad you came.",
};

function initials(name = 'Voice') {
  return name.split(/\s+/).slice(0, 2).map(word => word[0]).join('').toUpperCase();
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove('show'), 3200);
}

function updateWordCount() {
  const words = $('#scriptText').value.trim().match(/\S+/g)?.length || 0;
  $('#wordCount').textContent = `${words} word${words === 1 ? '' : 's'}`;
  const continuous = $('#speechMode')?.value === 'continuous';
  const chunks = Math.max(1, Math.ceil(words / (continuous ? 16 : 18)));
  const label = continuous ? 'FL2VA' : 'Ref2VA';
  $('#estimateText').textContent = `${chunks} linked ${label} chunk${chunks === 1 ? '' : 's'} · about ${chunks * 35}s render`;
}

function updateSpeechMode() {
  const continuous = $('#speechMode').value === 'continuous';
  $('#speechModeNote').textContent = continuous
    ? 'Experimental: carries latent audio context between chunks. Final output is still checked, but catalog personas are interpreted rather than cloned.'
    : 'Uses the selected reference on every chunk and rejects raw takes containing any words outside the script.';
  $('#contextSeconds').disabled = !continuous;
  updateWordCount();
}

function avatarClass(voice) {
  if (!voice.builtin) return 'custom';
  return voice.gender || 'nonbinary';
}

function selectVoice(voice) {
  state.selected = voice;
  $('#selectedName').textContent = voice.name;
  $('#selectedDescription').textContent = voice.description;
  $('#selectedAvatar').textContent = initials(voice.name);
  $('#selectedAvatar').className = `voice-avatar large ${avatarClass(voice)}`;
  renderVoices();
  closeVoicePanel();
}

function renderVoices() {
  const query = $('#voiceSearch').value.trim().toLowerCase();
  const visible = state.voices.filter(voice => {
    const filterMatch = state.filter === 'all' ||
      (state.filter === 'custom' ? !voice.builtin : voice.gender === state.filter);
    const haystack = [voice.name, voice.description, ...(voice.style_tags || [])].join(' ').toLowerCase();
    return filterMatch && haystack.includes(query);
  });
  $('#voiceCount').textContent = `${visible.length} of ${state.voices.length} voices`;
  $('#voiceList').replaceChildren(...visible.map(voice => {
    const button = document.createElement('button');
    button.className = `voice-item ${state.selected?.id === voice.id ? 'selected' : ''}`;
    const tags = (voice.style_tags || []).slice(0, 3).join(' · ') || voice.age_group || 'custom voice';
    button.innerHTML = `<span class="voice-avatar ${avatarClass(voice)}">${initials(voice.name)}</span><span><strong></strong><small></small></span><i class="ready-mark"></i>`;
    button.querySelector('strong').textContent = voice.name;
    button.querySelector('small').textContent = tags;
    button.querySelector('.ready-mark').style.opacity = voice.ready ? '.75' : '.15';
    button.addEventListener('click', () => selectVoice(voice));
    return button;
  }));
}

async function loadVoices(preferredId) {
  const response = await fetch('/v1/voices');
  if (!response.ok) throw new Error('Could not load the voice catalog.');
  state.voices = (await response.json()).data;
  const preferred = state.voices.find(v => v.id === preferredId) ||
    state.voices.find(v => v.id === state.selected?.id) ||
    state.voices.find(v => v.id === 'cinematic_ai_companion_f') || state.voices[0];
  if (preferred) selectVoice(preferred);
}

async function updateHealth() {
  try {
    const response = await fetch('/health', { cache: 'no-store' });
    const health = await response.json();
    const online = response.ok && health.ok && health.comfyui === 'online';
    const warmth = health.warmth || {};
    $('#systemPill').className = `system-pill ${online ? 'online' : 'offline'}`;
    $('#systemText').textContent = online ? (warmth.state === 'hot' ? 'H3 hot' : 'H3 ready') : 'ComfyUI offline';
    if ($('#residentState')) {
      $('#residentState').textContent = warmth.state === 'hot' ? `${warmth.gpu_reserved_gb} GB · hot` : (warmth.state || 'cold');
    }
    return health;
  } catch (_) {
    $('#systemPill').className = 'system-pill offline';
    $('#systemText').textContent = 'API offline';
    return null;
  }
}

function openVoicePanel() { $('#voicePanel').classList.add('open'); }
function closeVoicePanel() { $('#voicePanel').classList.remove('open'); }

function startProgress(kind = 'voice') {
  state.startedAt = Date.now();
  state.progress = 5;
  $('#progressBar').style.width = '5%';
  const titles = {
    voice: ['Finding the voice…', 'The first verified take usually takes around half a minute.'],
    music: ['Composing the arrangement…', 'H3 is laying out the complete stereo timeline. Longer tracks take proportionally longer.'],
    sfx: ['Sculpting the effect…', 'H3 is rendering a diegetic one-shot or bed with no speech and no score.'],
  };
  const later = {
    voice: ['Shaping the performance…', 'H3 is rendering and the CPU verifier is listening for every requested word.'],
    music: ['Building the track…', 'The instruments, sections, and marked lyrics are being rendered together as one continuous take.'],
    sfx: ['Settling the tail…', 'Attack, body, and decay are being shaped on the audio-first canvas.'],
  };
  $('#progressTitle').textContent = (titles[kind] || titles.voice)[0];
  $('#progressDetail').textContent = (titles[kind] || titles.voice)[1];
  $('#progressOverlay').classList.remove('hidden');
  clearInterval(state.timer);
  state.timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
    $('#elapsed').textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} elapsed`;
    state.progress = Math.min(92, state.progress + (state.progress < 55 ? 1.8 : .45));
    $('#progressBar').style.width = `${state.progress}%`;
    if (seconds > 25) {
      $('#progressTitle').textContent = (later[kind] || later.voice)[0];
      $('#progressDetail').textContent = (later[kind] || later.voice)[1];
    }
  }, 1000);
  clearInterval(state.queueTimer);
  state.queueTimer = setInterval(async () => {
    try {
      const queue = await (await fetch('/v1/queue', { cache: 'no-store' })).json();
      $('#queueText').textContent = queue.active ? 'Rendering on GPU' : queue.waiting ? `Queue: ${queue.waiting}` : 'Verifying take';
    } catch (_) { /* generation request remains authoritative */ }
  }, 2500);
}

function stopProgress() {
  clearInterval(state.timer);
  clearInterval(state.queueTimer);
  $('#progressBar').style.width = '100%';
  setTimeout(() => $('#progressOverlay').classList.add('hidden'), 280);
}

function addHistory(entry) {
  state.history.unshift(entry);
  $('#historySection').classList.remove('hidden');
  const list = $('#historyList');
  list.replaceChildren(...state.history.slice(0, 8).map((item, index) => {
    const row = document.createElement('div');
    row.className = 'history-item';
    row.innerHTML = `<button class="history-play" aria-label="Play">▶</button><span><strong></strong><small></small></span><a download>Save</a>`;
    row.querySelector('strong').textContent = item.title;
    row.querySelector('small').textContent = `${item.detail} · ${item.format.toUpperCase()} · take ${state.history.length - index}`;
    row.querySelector('a').href = item.url;
    row.querySelector('a').download = item.filename;
    row.querySelector('.history-play').addEventListener('click', () => {
      $('#audioPlayer').src = item.url;
      $('#resultTitle').textContent = item.title;
      $('#resultCard').classList.remove('hidden');
      state.latestPromptJob = item.promptJob || null;
      state.latestPromptKind = item.promptKind || null;
      $('#viewPromptButton').classList.toggle('hidden', !state.latestPromptJob);
      $('#audioPlayer').play();
      $('#resultCard').scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
    return row;
  }));
}

async function generate() {
  if (state.generating) return;
  const text = $('#scriptText').value.trim();
  if (!text) return showToast('Write something for the voice to say.');
  const description = $('#voiceDescription').value.trim();
  if (state.mode === 'described' && !description) return showToast('Describe the voice you want to create.');
  if (state.mode === 'catalog' && !state.selected) return showToast('Choose a voice first.');

  state.generating = true;
  $('#generateButton').disabled = true;
  $('#generateLabel').textContent = 'Generating…';
  startProgress();
  const format = $('#format').value;
  const speechMode = $('#speechMode').value;
  const payload = {
    model: speechMode === 'continuous' ? 'minimax-h3-fl2va-continuous' : 'minimax-h3-ref2va',
    speech_mode: speechMode, input: text,
    voice: state.mode === 'described' ? 'described_voice' : state.selected.id,
    response_format: format, stream: false,
    instructions: $('#direction').value.trim(),
    emotion: $('#emotion').value,
    emotion_intensity: Number($('#intensity').value) / 100,
    spatial_preset: $('#space').value,
    continuity_context_seconds: Number($('#contextSeconds').value),
    steps: Number($('#steps').value), resolution: Number($('#resolution').value),
    channels: 2, verify: $('#verify').checked,
    fidelity_mode: $('#verify').checked ? 'exact' : 'unverified',
    candidate_pool_size: Number($('#candidatePool').value),
    generation_padding_seconds: Number($('#generationPadding').value),
    audio_ref_keep: Number($('#audioRefKeep').value),
    target_words: speechMode === 'continuous' ? 16 : 18,
    max_words: speechMode === 'continuous' ? 20 : 24,
    strict_fidelity: speechMode !== 'continuous',
  };
  if (state.mode === 'described') payload.voice_description = description;

  try {
    const response = await fetch('/v1/audio/speech', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let detail = `Generation failed (${response.status})`;
      try { detail = (await response.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const voiceName = state.mode === 'described' ? 'Described voice' : state.selected.name;
    const filename = `vox-${voiceName.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${Date.now()}.${format}`;
    $('#audioPlayer').src = url;
    $('#resultTitle').textContent = voiceName;
    const renderTime = Number(response.headers.get('x-h3-render-seconds') || 0);
    const modeLabel = response.headers.get('x-voice-mode') === 'fl2va-motion-context' ? 'continuous FL2VA' : 'Ref2VA';
    const similarity = Number(response.headers.get('x-asr-min-similarity') || 0);
    const leadingTrim = Number(response.headers.get('x-voice-leading-trim-seconds') || 0);
    $('#resultMeta').textContent = `${format.toUpperCase()} · stereo · ${modeLabel} · ${response.headers.get('x-voice-chunks') || '?'} chunk(s)${leadingTrim ? ` · ${leadingTrim.toFixed(2)}s opening cleaned` : ''}${similarity ? ` · ${Math.round(similarity * 100)}% minimum transcript match` : ''}${renderTime ? ` · ${renderTime.toFixed(1)}s H3 render` : ''}`;
    $('#downloadButton').href = url;
    $('#downloadButton').download = filename;
    state.latestPromptJob = null;
    $('#viewPromptButton').classList.add('hidden');
    $('#resultCard').classList.remove('hidden');
    addHistory({ url, filename, title: voiceName, detail: `${text.split(/\s+/).length} spoken words`, format });
    stopProgress();
    $('#audioPlayer').play().catch(() => {});
    $('#resultCard').scrollIntoView({ behavior: 'smooth', block: 'center' });
    showToast('Performance ready.');
    if (state.mode === 'described') await loadVoices(response.headers.get('x-voice-id'));
  } catch (error) {
    stopProgress();
    showToast(error.message || 'Generation failed.');
  } finally {
    state.generating = false;
    $('#generateButton').disabled = false;
    $('#generateLabel').textContent = 'Generate performance';
  }
}

function setCreation(kind) {
  state.creation = kind;
  $$('#creationSwitch button').forEach(button => button.classList.toggle('active', button.dataset.creation === kind));
  $('#voiceStudio').classList.toggle('hidden', kind !== 'voice');
  $('#musicStudio').classList.toggle('hidden', kind !== 'music');
  $('#sfxStudio').classList.toggle('hidden', kind !== 'sfx');
  $('.workspace').classList.toggle('music-mode', kind === 'music' || kind === 'sfx');
  const copy = {
    voice: {
      eyebrow: 'Performance desk',
      title: 'Give the words<br><em>a pulse.</em>',
      intro: 'Direct a performance, place it in a room, and let H3 build a voice that feels present.',
    },
    music: {
      eyebrow: 'Music laboratory',
      title: 'Shape a song<br><em>from silence.</em>',
      intro: 'Design the sound, arrange the sections, and mark only the words H3 should sing.',
    },
    sfx: {
      eyebrow: 'Effects bay',
      title: 'Cut a hit<br><em>without a score.</em>',
      intro: 'Describe a diegetic effect or bed. H3 stays silent on speech and non-diegetic music.',
    },
  }[kind] || {
    eyebrow: 'Performance desk',
    title: 'Give the words<br><em>a pulse.</em>',
    intro: 'Direct a performance, place it in a room, and let H3 build a voice that feels present.',
  };
  $('.studio-intro .eyebrow').textContent = copy.eyebrow;
  $('.studio-intro h1').innerHTML = copy.title;
  $('.intro-copy').textContent = copy.intro;
  if (kind === 'music' || kind === 'sfx') closeVoicePanel();
  if (kind === 'sfx') updateSfxPlan();
}

function setMusicKind(kind) {
  state.musicKind = kind;
  $$('.music-kind-tab').forEach(button => button.classList.toggle('active', button.dataset.kind === kind));
  $('#songFields').classList.toggle('hidden', kind !== 'song');
  $('#generateMusicLabel').textContent = kind === 'song' ? 'Compose song' : 'Compose track';
  updateMusicPlan();
}

function snappedMusicDuration(seconds) {
  let frames = Math.max(5, Math.round(seconds * 24));
  while (frames % 17 !== 5) frames += 1;
  return frames / 24;
}

function lyricLineCount() {
  return lyricStats().lines;
}

function lyricStats() {
  let lines = 0;
  let words = 0;
  let vocalSections = 0;
  let instrumentalSections = 0;
  let sectionLines = 0;
  let sawSection = false;
  for (const raw of $('#musicLyrics').value.split(/\n/)) {
    const line = raw.trim();
    if (/^\[.+\]$/.test(line)) {
      if (sawSection) {
        if (sectionLines) vocalSections += 1;
        else instrumentalSections += 1;
      }
      sawSection = true;
      sectionLines = 0;
    } else if (line) {
      lines += 1;
      words += line.split(/\s+/).length;
      sectionLines += 1;
    }
  }
  if (sawSection) {
    if (sectionLines) vocalSections += 1;
    else instrumentalSections += 1;
  } else if (sectionLines) vocalSections = 1;
  return { lines, words, vocalSections, instrumentalSections };
}

function recommendedMusicDuration(stats) {
  const style = $('#musicStyle').value.toLowerCase();
  const match = style.match(/\b(\d{2,3}(?:\.\d+)?)\s*bpm\b/i);
  const bpm = Math.min(220, Math.max(45, match ? Number(match[1]) : 105));
  const rap = /\b(rap|hip-hop|hip hop|trap|drill|spoken flow)\b/.test(style);
  const fast = /(fast-paced|fast paced|double-time|double time|punk|drum and bass|\bdnb\b|hyperpop)/.test(style);
  const slow = /(slow|ballad|lullaby|sparse|drawn-out|drawn out)/.test(style) && !rap;
  let wordsPerSecond = 2.15 + (bpm - 100) * 0.006 + (rap ? 0.65 : 0) + (fast ? 0.20 : 0) - (slow ? 0.22 : 0);
  wordsPerSecond = Math.min(3.45, Math.max(1.75, wordsPerSecond));
  const averageLineWords = stats.words / Math.max(1, stats.lines);
  const linePause = rap || fast ? 0.04 + Math.min(8, averageLineWords) * 0.01 : slow ? 0.12 + Math.min(8, averageLineWords) * 0.02 : 0.08 + Math.min(8, averageLineWords) * 0.02;
  const raw = stats.words / wordsPerSecond + stats.lines * linePause + stats.instrumentalSections * 2 + Math.max(0, stats.vocalSections - 1) * 0.6 + 1;
  const recommended = Math.max(10, Math.min(60, 5 * Math.floor(raw / 5 + 0.5)));
  return { seconds: recommended, raw, bpm, wordsPerSecond, capped: raw > 60, fast: rap || fast };
}

function updateMusicPlan() {
  const stats = lyricStats();
  const recommendation = recommendedMusicDuration(stats);
  const auto = state.musicKind === 'song' && $('#musicAutoDuration').checked && stats.lines;
  $('#musicDuration').disabled = Boolean(auto);
  if (auto) $('#musicDuration').value = recommendation.seconds;
  const requested = Number($('#musicDuration').value);
  const duration = snappedMusicDuration(requested);
  $('#musicDurationValue').textContent = `${requested} second${requested === 1 ? '' : 's'}${auto ? ' · auto' : ''}`;
  $('#musicDuration').style.background = `linear-gradient(90deg, var(--paper) ${(requested - 5) / 55 * 100}%, #393832 ${(requested - 5) / 55 * 100}%)`;
  const lines = stats.lines;
  const chips = [`${duration.toFixed(1)} seconds`, '32×32 audio-first canvas', 'No video decode'];
  let warning = '';
  if (state.musicKind === 'song') {
    const perLine = duration / Math.max(1, lines);
    chips.splice(1, 0, `${lines} lyric lines`, `${stats.vocalSections} vocal sections`, `${recommendation.bpm} BPM estimate`, `${perLine.toFixed(1)}s per line`);
    if (stats.vocalSections > 4 || lines > 18) warning = 'Outside H3’s proven zone: aim for 3–4 vocal sections and 8–14 complete lines for the most natural song.';
    else if (auto && recommendation.capped) warning = `The lyric wants about ${Math.ceil(recommendation.raw)}s in this style, but local H3 is capped at 60s; expect denser phrasing.`;
    else if (lines && perLine < 2.4) warning = 'Dense lyric timing—the vocal may become clipped, rushed, or lose its ending.';
    else if (stats.words / duration > 1.9) warning = 'High sung word density—add time or shorten the lyric for more natural phrasing.';
    else if (lines && perLine > 5.5) warning = 'Very open timing—expect instrumental vamps between lyric lines.';
    $('#lyricTiming').textContent = lines ? `${lines} lines · ${stats.words} words · ${stats.vocalSections} vocal sections` : 'Add lyric lines';
  }
  if (warning) chips.push(warning);
  $('#songPlan').replaceChildren(...chips.map(text => {
    const span = document.createElement('span');
    span.textContent = text;
    if (text === warning) span.className = 'warning';
    return span;
  }));
  $('#musicEstimate').textContent = `${state.musicKind === 'song' ? 'Full song' : 'Instrumental'} · ${duration.toFixed(1)}s stereo · approximately ${Math.max(1, Math.round(duration / 15)) * 35}s+ render`;
}

function renderMusicPresets() {
  const query = state.musicPresetQuery.trim().toLowerCase();
  const visible = state.musicPresets.filter(preset => {
    const categoryMatch = state.musicCategory === 'all' || preset.category === state.musicCategory;
    const haystack = `${preset.name} ${preset.category} ${preset.style} ${preset.instrumentation} ${preset.vocalist || ''}`.toLowerCase();
    return categoryMatch && (!query || haystack.includes(query));
  });
  $('#musicPresetCount').textContent = `${visible.length} of ${state.musicPresets.length}`;
  const cards = visible.map(preset => {
    const button = document.createElement('button');
    button.className = `music-preset ${state.musicPreset === preset.id ? 'active' : ''}`;
    button.innerHTML = '<span></span><strong></strong><small></small>';
    button.querySelector('span').textContent = preset.category;
    button.querySelector('strong').textContent = preset.name;
    button.querySelector('small').textContent = preset.style;
    button.addEventListener('click', () => {
      state.musicPreset = preset.id;
      $('#musicStyle').value = preset.style;
      $('#musicInstrumentation').value = preset.instrumentation;
      if (preset.vocalist) $('#musicVocalist').value = preset.vocalist;
      renderMusicPresets();
      updateMusicPlan();
    });
    return button;
  });
  if (!cards.length) {
    const empty = document.createElement('div');
    empty.className = 'music-preset-empty';
    empty.textContent = 'No matching sounds. Try a broader search.';
    cards.push(empty);
  }
  $('#musicPresets').replaceChildren(...cards);
}

async function loadMusicPresets() {
  const response = await fetch('/v1/audio/music/presets');
  if (!response.ok) throw new Error('Could not load music presets.');
  state.musicPresets = (await response.json()).data;
  const categories = [...new Set(state.musicPresets.map(preset => preset.category))];
  $('#musicCategory').replaceChildren(
    Object.assign(document.createElement('option'), { value: 'all', textContent: 'All styles' }),
    ...categories.map(category => Object.assign(document.createElement('option'), { value: category, textContent: category }))
  );
  renderMusicPresets();
}

async function generateMusic() {
  if (state.generating) return;
  const style = $('#musicStyle').value.trim();
  const instrumentation = $('#musicInstrumentation').value.trim();
  const lyrics = $('#musicLyrics').value.trim();
  if (!style || !instrumentation) return showToast('Describe the style and instrumentation first.');
  if (state.musicKind === 'song' && !lyricLineCount()) return showToast('Add at least one lyric line to the song.');
  state.generating = true;
  $('#generateMusicButton').disabled = true;
  $('#generateMusicLabel').textContent = 'Composing…';
  startProgress('music');
  const format = $('#musicFormat').value;
  const seedText = $('#musicSeed').value.trim();
  const payload = {
    mode: state.musicKind, title: $('#musicTitle').value.trim() || 'H3 music',
    preset: state.musicPreset, style, instrumentation,
    vocalist: $('#musicVocalist').value.trim(), lyrics,
    scene: $('#musicScene').value.trim(),
    soundscape: $('#musicSoundscape').value.trim() || 'N/A',
    duration_seconds: Number($('#musicDuration').value),
    duration_mode: state.musicKind === 'song' && $('#musicAutoDuration').checked ? 'auto' : 'manual',
    prompt_profile: $('#musicPromptProfile').value,
    steps: Number($('#musicSteps').value), response_format: format,
  };
  if (seedText) payload.seed = Number(seedText);
  try {
    const response = await fetch('/v1/audio/music', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let detail = `Music generation failed (${response.status})`;
      try { detail = (await response.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const title = payload.title;
    const filename = `h3-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${Date.now()}.${format}`;
    const duration = Number(response.headers.get('x-h3-duration-seconds') || payload.duration_seconds);
    const renderTime = Number(response.headers.get('x-h3-render-seconds') || 0);
    const promptJob = response.headers.get('x-h3-job-id');
    const generatedSeed = response.headers.get('x-h3-seed');
    const durationMode = response.headers.get('x-h3-duration-mode');
    $('#audioPlayer').src = url;
    $('#resultTitle').textContent = title;
    $('#resultMeta').textContent = `${state.musicKind === 'song' ? 'Song' : 'Instrumental'} · ${duration.toFixed(1)}s${durationMode === 'auto' ? ' auto-fit' : ''} · ${format.toUpperCase()} stereo${generatedSeed ? ` · seed ${generatedSeed}` : ''}${renderTime ? ` · ${renderTime.toFixed(1)}s render` : ''}`;
    $('#downloadButton').href = url;
    $('#downloadButton').download = filename;
    state.latestPromptJob = promptJob;
    state.latestPromptKind = 'music';
    $('#viewPromptButton').classList.toggle('hidden', !promptJob);
    $('#resultCard').classList.remove('hidden');
    addHistory({ url, filename, title, detail: `${state.musicKind} · ${duration.toFixed(1)} seconds`, format, promptJob, promptKind: 'music' });
    stopProgress();
    $('#audioPlayer').play().catch(() => {});
    $('#resultCard').scrollIntoView({ behavior: 'smooth', block: 'center' });
    showToast(`${state.musicKind === 'song' ? 'Song' : 'Track'} ready.`);
  } catch (error) {
    stopProgress();
    showToast(error.message || 'Music generation failed.');
  } finally {
    state.generating = false;
    $('#generateMusicButton').disabled = false;
    $('#generateMusicLabel').textContent = state.musicKind === 'song' ? 'Compose song' : 'Compose track';
  }
}

function setSfxKind(kind) {
  state.sfxKind = kind;
  $$('.sfx-kind-tab').forEach(button => button.classList.toggle('active', button.dataset.kind === kind));
  $('#generateSfxLabel').textContent = kind === 'loop_bed' ? 'Render bed' : 'Render effect';
  updateSfxPlan();
}

function snappedSfxDuration(seconds) {
  let frames = Math.max(5, Math.round(seconds * 24));
  const lower = Math.max(5, frames - ((frames - 5) % 17));
  const upper = lower + 17;
  frames = (frames - lower <= upper - frames) ? lower : upper;
  return frames / 24;
}

function recommendedSfxDuration(description, kind) {
  const text = description.toLowerCase();
  const words = description.trim().match(/\S+/g)?.length || 0;
  const multi = /(several|multiple|sequence|then |followed by|walk|footsteps|pass-by|pass by|open and close|cycle|pattern)/.test(text);
  const longish = /(roll|decay|settle|sustain|hold|continuous|steady|bed|ambience|atmosphere|loop)/.test(text);
  const tiny = /(click|tick|blip|snap|tap|ding|shutter|toggle|single)/.test(text) && !multi;
  let raw;
  let recommended;
  if (kind === 'loop_bed') {
    raw = 12 + (longish ? 2 : 0) + Math.min(6, words * 0.05);
    recommended = Math.max(6, Math.min(30, Math.floor(raw + 0.5)));
  } else {
    raw = 3;
    if (tiny) raw = 2;
    if (multi) raw = Math.max(raw, 5);
    if (longish) raw = Math.max(raw, 5);
    if (words > 24) raw += 1.5;
    recommended = Math.max(1.5, Math.min(30, 0.5 * Math.floor(raw * 2 + 0.5)));
  }
  return { seconds: recommended, raw, multi, tiny };
}

function updateSfxPlan() {
  const description = $('#sfxDescription').value.trim();
  const kind = state.sfxKind;
  const recommendation = recommendedSfxDuration(description, kind);
  const preset = state.sfxPresets.find(item => item.id === state.sfxPreset);
  const auto = $('#sfxAutoDuration').checked;
  $('#sfxDuration').disabled = Boolean(auto);
  if (auto) {
    if (preset && description === preset.description && preset.default_seconds) {
      $('#sfxDuration').value = preset.default_seconds;
    } else {
      $('#sfxDuration').value = recommendation.seconds;
    }
  }
  const requested = Number($('#sfxDuration').value);
  const duration = snappedSfxDuration(requested);
  const pct = (requested - 1.5) / (30 - 1.5) * 100;
  $('#sfxDurationValue').textContent = `${requested} second${requested === 1 ? '' : 's'}${auto ? ' · auto' : ''}`;
  $('#sfxDuration').style.background = `linear-gradient(90deg, var(--paper) ${pct}%, #393832 ${pct}%)`;
  const chips = [
    `${duration.toFixed(1)} seconds`,
    kind === 'loop_bed' ? 'Loop / bed' : 'One-shot',
    '32×32 audio-first canvas',
    'No speech · no score',
  ];
  let warning = '';
  if (kind === 'oneshot' && requested > 12) warning = 'Long one-shots can fill with extra events—prefer loop/bed for continuous texture.';
  if (kind === 'loop_bed' && requested < 6) warning = 'Beds usually need at least ~6s so the texture can establish.';
  if (warning) chips.push(warning);
  $('#sfxPlan').replaceChildren(...chips.map(text => {
    const span = document.createElement('span');
    span.textContent = text;
    if (text === warning) span.className = 'warning';
    return span;
  }));
  $('#sfxEstimate').textContent = `${kind === 'loop_bed' ? 'Bed' : 'One-shot'} · ${duration.toFixed(1)}s stereo · approximately ${Math.max(1, Math.round(duration / 8)) * 25}s+ render`;
}

function renderSfxPresets() {
  const query = state.sfxPresetQuery.trim().toLowerCase();
  const visible = state.sfxPresets.filter(preset => {
    const categoryMatch = state.sfxCategory === 'all' || preset.category === state.sfxCategory;
    const haystack = `${preset.name} ${preset.category} ${preset.description} ${preset.kind}`.toLowerCase();
    return categoryMatch && (!query || haystack.includes(query));
  });
  $('#sfxPresetCount').textContent = `${visible.length} of ${state.sfxPresets.length}`;
  const cards = visible.map(preset => {
    const button = document.createElement('button');
    button.className = `music-preset ${state.sfxPreset === preset.id ? 'active' : ''}`;
    button.innerHTML = '<span></span><strong></strong><small></small>';
    button.querySelector('span').textContent = `${preset.category} · ${preset.kind}`;
    button.querySelector('strong').textContent = preset.name;
    button.querySelector('small').textContent = preset.description;
    button.addEventListener('click', () => {
      state.sfxPreset = preset.id;
      $('#sfxDescription').value = preset.description;
      if (preset.space) $('#sfxSpace').value = preset.space;
      if (preset.kind) setSfxKind(preset.kind);
      if (preset.default_seconds) $('#sfxDuration').value = preset.default_seconds;
      renderSfxPresets();
      updateSfxPlan();
    });
    return button;
  });
  if (!cards.length) {
    const empty = document.createElement('div');
    empty.className = 'music-preset-empty';
    empty.textContent = 'No matching effects. Try a broader search.';
    cards.push(empty);
  }
  $('#sfxPresets').replaceChildren(...cards);
}

async function loadSfxPresets() {
  const response = await fetch('/v1/audio/sfx/presets');
  if (!response.ok) throw new Error('Could not load SFX presets.');
  state.sfxPresets = (await response.json()).data;
  const categories = [...new Set(state.sfxPresets.map(preset => preset.category))];
  $('#sfxCategory').replaceChildren(
    Object.assign(document.createElement('option'), { value: 'all', textContent: 'All categories' }),
    ...categories.map(category => Object.assign(document.createElement('option'), { value: category, textContent: category }))
  );
  renderSfxPresets();
  updateSfxPlan();
}

async function generateSfx() {
  if (state.generating) return;
  const description = $('#sfxDescription').value.trim();
  if (!description) return showToast('Describe the effect first.');
  state.generating = true;
  $('#generateSfxButton').disabled = true;
  $('#generateSfxLabel').textContent = 'Rendering…';
  startProgress('sfx');
  const format = $('#sfxFormat').value;
  const seedText = $('#sfxSeed').value.trim();
  const payload = {
    title: $('#sfxTitle').value.trim() || 'H3 sfx',
    preset: state.sfxPreset,
    description,
    kind: state.sfxKind,
    space: $('#sfxSpace').value.trim(),
    intensity: Number($('#sfxIntensity').value) / 100,
    duration_seconds: Number($('#sfxDuration').value),
    duration_mode: $('#sfxAutoDuration').checked ? 'auto' : 'manual',
    prompt_profile: 'diegetic_v1',
    steps: Number($('#sfxSteps').value),
    resolution: Number($('#sfxResolution').value),
    response_format: format,
  };
  if (seedText) payload.seed = Number(seedText);
  try {
    const response = await fetch('/v1/audio/sfx', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let detail = `SFX generation failed (${response.status})`;
      try { detail = (await response.json()).detail || detail; } catch (_) {}
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const title = payload.title;
    const filename = `h3-sfx-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${Date.now()}.${format}`;
    const duration = Number(response.headers.get('x-h3-duration-seconds') || payload.duration_seconds);
    const renderTime = Number(response.headers.get('x-h3-render-seconds') || 0);
    const promptJob = response.headers.get('x-h3-job-id');
    const generatedSeed = response.headers.get('x-h3-seed');
    const sfxKind = response.headers.get('x-h3-sfx-kind') || state.sfxKind;
    $('#audioPlayer').src = url;
    $('#resultTitle').textContent = title;
    $('#resultMeta').textContent = `SFX · ${sfxKind} · ${duration.toFixed(1)}s · ${format.toUpperCase()} stereo${generatedSeed ? ` · seed ${generatedSeed}` : ''}${renderTime ? ` · ${renderTime.toFixed(1)}s render` : ''}`;
    $('#downloadButton').href = url;
    $('#downloadButton').download = filename;
    state.latestPromptJob = promptJob;
    state.latestPromptKind = 'sfx';
    $('#viewPromptButton').classList.toggle('hidden', !promptJob);
    $('#resultCard').classList.remove('hidden');
    addHistory({ url, filename, title, detail: `sfx · ${sfxKind} · ${duration.toFixed(1)} seconds`, format, promptJob, promptKind: 'sfx' });
    stopProgress();
    $('#audioPlayer').play().catch(() => {});
    $('#resultCard').scrollIntoView({ behavior: 'smooth', block: 'center' });
    showToast('Effect ready.');
  } catch (error) {
    stopProgress();
    showToast(error.message || 'SFX generation failed.');
  } finally {
    state.generating = false;
    $('#generateSfxButton').disabled = false;
    $('#generateSfxLabel').textContent = state.sfxKind === 'loop_bed' ? 'Render bed' : 'Render effect';
  }
}

async function showFinalPrompt() {
  if (!state.latestPromptJob) return;
  const dialog = $('#promptDialog');
  $('#finalPromptText').textContent = 'Loading…';
  dialog.showModal();
  const kind = state.latestPromptKind === 'sfx' ? 'sfx' : 'music';
  try {
    const response = await fetch(`/v1/audio/${kind}/jobs/${state.latestPromptJob}/prompt`);
    if (!response.ok) throw new Error('Could not load the stored generation prompt.');
    const result = await response.json();
    $('#finalPromptText').textContent = result.prompt;
  } catch (error) {
    $('#finalPromptText').textContent = error.message;
  }
}

async function copyFinalPrompt() {
  const text = $('#finalPromptText').textContent;
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
  } else {
    const helper = document.createElement('textarea');
    helper.value = text;
    helper.setAttribute('readonly', '');
    helper.style.position = 'fixed';
    helper.style.opacity = '0';
    document.body.appendChild(helper);
    helper.select();
    document.execCommand('copy');
    helper.remove();
  }
  showToast('Final prompt copied.');
}

async function enroll(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector('button[type=submit]');
  const message = $('#enrollMessage');
  button.disabled = true;
  button.textContent = 'Preparing reference…';
  message.textContent = 'Extracting and transcribing the voice on CPU.';
  try {
    const response = await fetch('/v1/voices', { method: 'POST', body: new FormData(form) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Voice upload failed.');
    await loadVoices(data.id);
    $('#enrollDialog').close();
    form.reset();
    showToast(`${data.name} is ready to use.`);
  } catch (error) {
    message.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = 'Prepare voice';
  }
}

function bindEvents() {
  $('#voiceSearch').addEventListener('input', renderVoices);
  $('#scriptText').addEventListener('input', updateWordCount);
  $('#speechMode').addEventListener('change', updateSpeechMode);
  $('#intensity').addEventListener('input', event => {
    $('#intensityValue').textContent = `${event.target.value}%`;
    event.target.style.background = `linear-gradient(90deg, var(--paper) ${event.target.value}%, #393832 ${event.target.value}%)`;
  });
  $$('#voiceFilters .filter').forEach(button => button.addEventListener('click', () => {
    $$('#voiceFilters .filter').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    state.filter = button.dataset.filter;
    renderVoices();
  }));
  $$('.composer-tab').forEach(button => button.addEventListener('click', () => {
    $$('.composer-tab').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    state.mode = button.dataset.mode;
    $('#describedBox').classList.toggle('hidden', state.mode !== 'described');
    $('#selectedVoiceCard').classList.toggle('hidden', state.mode === 'described');
  }));
  $$('.prompt-chip').forEach(button => button.addEventListener('click', () => {
    $('#scriptText').value = prompts[button.dataset.prompt];
    updateWordCount();
    if (button.dataset.prompt === 'playful') $('#emotion').value = 'playful';
    if (button.dataset.prompt === 'dramatic') $('#emotion').value = 'authoritative';
  }));
  $('#openVoices').addEventListener('click', openVoicePanel);
  $('#changeVoice').addEventListener('click', openVoicePanel);
  $('#closeVoices').addEventListener('click', closeVoicePanel);
  $('#generateButton').addEventListener('click', generate);
  $$('#creationSwitch button').forEach(button => button.addEventListener('click', () => setCreation(button.dataset.creation)));
  $$('.music-kind-tab').forEach(button => button.addEventListener('click', () => setMusicKind(button.dataset.kind)));
  $$('.sfx-kind-tab').forEach(button => button.addEventListener('click', () => setSfxKind(button.dataset.kind)));
  $('#musicDuration').addEventListener('input', updateMusicPlan);
  $('#musicAutoDuration').addEventListener('change', updateMusicPlan);
  $('#musicStyle').addEventListener('input', updateMusicPlan);
  $('#musicLyrics').addEventListener('input', updateMusicPlan);
  $('#generateMusicButton').addEventListener('click', generateMusic);
  $('#generateSfxButton').addEventListener('click', generateSfx);
  $('#viewPromptButton').addEventListener('click', showFinalPrompt);
  $('#copyPromptButton').addEventListener('click', copyFinalPrompt);
  $('#clearMusicPreset').addEventListener('click', () => { state.musicPreset = null; renderMusicPresets(); });
  $('#clearSfxPreset').addEventListener('click', () => { state.sfxPreset = null; renderSfxPresets(); });
  $('#musicPresetSearch').addEventListener('input', event => {
    state.musicPresetQuery = event.target.value;
    renderMusicPresets();
  });
  $('#musicCategory').addEventListener('change', event => {
    state.musicCategory = event.target.value;
    renderMusicPresets();
  });
  $('#sfxPresetSearch').addEventListener('input', event => {
    state.sfxPresetQuery = event.target.value;
    renderSfxPresets();
  });
  $('#sfxCategory').addEventListener('change', event => {
    state.sfxCategory = event.target.value;
    renderSfxPresets();
  });
  $('#sfxDuration').addEventListener('input', updateSfxPlan);
  $('#sfxAutoDuration').addEventListener('change', updateSfxPlan);
  $('#sfxDescription').addEventListener('input', updateSfxPlan);
  $('#sfxIntensity').addEventListener('input', event => {
    $('#sfxIntensityValue').textContent = `${event.target.value}%`;
    event.target.style.background = `linear-gradient(90deg, var(--paper) ${event.target.value}%, #393832 ${event.target.value}%)`;
  });
  $$('.section-tools button').forEach(button => button.addEventListener('click', () => {
    const editor = $('#musicLyrics');
    const addition = `\n\n[${button.dataset.section}]\n`;
    const start = editor.selectionStart;
    editor.setRangeText(addition, start, editor.selectionEnd, 'end');
    editor.focus();
    updateMusicPlan();
  }));
  $('#hideProgress').addEventListener('click', () => $('#progressOverlay').classList.add('hidden'));
  $('#openEnroll').addEventListener('click', () => $('#enrollDialog').showModal());
  $('#closeEnroll').addEventListener('click', () => $('#enrollDialog').close());
  $('#enrollForm').addEventListener('submit', enroll);
  $('.upload-zone input').addEventListener('change', event => {
    const file = event.target.files[0];
    if (file) $('.upload-zone strong').textContent = file.name;
  });
}

async function init() {
  bindEvents();
  updateWordCount();
  updateMusicPlan();
  updateSfxPlan();
  $('#sfxIntensity').dispatchEvent(new Event('input'));
  if (location.hash === '#music' || location.hash === '#song') setCreation('music');
  if (location.hash === '#song') setMusicKind('song');
  if (location.hash === '#sfx' || location.hash === '#foley') setCreation('sfx');
  try {
    const [_, emotions] = await Promise.all([
      loadVoices(),
      fetch('/v1/audio/emotions').then(r => r.json()),
      loadMusicPresets(),
      loadSfxPresets(),
    ]);
    $('#emotion').replaceChildren(...emotions.data.map(emotion => {
      const option = document.createElement('option');
      option.value = emotion;
      option.textContent = emotion[0].toUpperCase() + emotion.slice(1);
      if (emotion === 'tender') option.selected = true;
      return option;
    }));
  } catch (error) { showToast(error.message); }
  await updateHealth();
  setInterval(updateHealth, 15000);
}

init();
