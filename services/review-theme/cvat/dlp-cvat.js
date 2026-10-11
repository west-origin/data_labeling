// aniatman 검수 화면 테마 (CVAT). 공식 UI 위에 얹는 스크립트다 (ADR 0038). 화면 소스는 고치지 않는다.
// 하는 일:
//   1. 다크/화이트 모드: <html data-dlp-mode>를 정한다. 검수자가 고른 값은 localStorage(dlp-mode)에
//      기억하고, 고른 적이 없으면 dlp-tokens.css의 --dlp-default-mode(config/review_ui.yaml)를 쓴다.
//   2. 머리글 오른쪽에 모드 전환 버튼을 넣는다 (React가 머리글을 다시 그리면 다시 넣는다).
//   3. 작업 화면(/tasks/<작업>/jobs/<job>)에서 프로젝트 이름으로 검수 단계를 알아내 머리글에 배지를 단다:
//      블러 검수(원본 영상, 권한자 전용) / 공간 라벨 검수(블러본). 배지는 안내일 뿐 권한 검사가 아니다
//      (권한은 dlp review create의 담당자 검사와 CVAT 배정이 정한다).
(function () {
  'use strict';

  var KEY = 'dlp-mode';
  var root = document.documentElement;
  // 프로젝트 이름 접두사 → 배지. dlp_review.tasks의 PRIVACY_PROJECT·SPATIAL_PROJECT와 같아야 한다
  // (프로젝트 이름은 ontology_projects.ontology_project가 온톨로지 버전을 덧붙인다).
  // 테스트(test_ui_theme)가 두 상수와 이 표의 일치를 검사한다.
  var STAGES = [
    { prefix: 'dlp-privacy', text: '원본 영상 · 권한자 전용', kind: 'danger', title: '블러 검수' },
    { prefix: 'dlp-labeling-spatial', text: '블러본 · 원본 아님', kind: 'muted', title: '공간 라벨 검수' },
  ];

  function stored() {
    try {
      var v = window.localStorage.getItem(KEY);
      return v === 'dark' || v === 'light' ? v : null;
    } catch (e) {
      return null;
    }
  }

  function defaultMode() {
    var v = window.getComputedStyle(root).getPropertyValue('--dlp-default-mode').trim();
    return v === 'light' ? 'light' : 'dark';
  }

  function current() {
    return root.getAttribute('data-dlp-mode') === 'light' ? 'light' : 'dark';
  }

  function setMode(mode, remember) {
    root.setAttribute('data-dlp-mode', mode);
    if (remember) {
      try {
        window.localStorage.setItem(KEY, mode);
      } catch (e) {
        // 저장할 수 없는 브라우저(사생활 보호 모드 등)에서는 이번 화면에만 적용한다
      }
    }
    var btn = document.querySelector('.dlp-mode-toggle');
    if (btn) label(btn);
  }

  function label(btn) {
    var dark = current() === 'dark';
    btn.textContent = dark ? '☀ 화이트 모드' : '☾ 다크 모드';
    btn.setAttribute('title', dark ? '화이트 모드로 전환' : '다크 모드로 전환');
  }

  // 머리글 그리기 전에 깜빡이지 않게 기억한 값(없으면 다크)을 바로 적용하고,
  // 테마 CSS가 읽힌 뒤 기억한 값이 없으면 설정의 처음 모드로 맞춘다.
  setMode(stored() || 'dark', false);
  window.addEventListener('load', function () {
    if (!stored()) setMode(defaultMode(), false);
  });

  function ensureToggle() {
    var host = document.querySelector('.cvat-right-header');
    if (!host || host.querySelector('.dlp-mode-toggle')) return;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'dlp-mode-toggle';
    label(btn);
    btn.addEventListener('click', function () {
      setMode(current() === 'dark' ? 'light' : 'dark', true);
    });
    host.insertBefore(btn, host.firstChild);
  }

  // 작업 화면의 검수 단계 배지. 같은 작업이면 다시 묻지 않는다.
  var badgeTask = null;
  var badgeStage = null;

  function jobTaskId() {
    var m = window.location.pathname.match(/^\/tasks\/(\d+)\/jobs\/\d+/);
    return m ? m[1] : null;
  }

  function getJson(url) {
    // Accept 헤더를 직접 정하지 않는다 (CVAT API는 application/json만 적으면 406으로 거절한다)
    return window.fetch(url, { credentials: 'same-origin' })
      .then(function (r) {
        if (!r.ok) throw new Error(url + ' ' + r.status);
        return r.json();
      });
  }

  function lookupStage(taskId) {
    badgeTask = taskId;
    badgeStage = null;
    // 작업 응답에 프로젝트 이름(project_name)이 함께 온다
    getJson('/api/tasks/' + taskId)
      .then(function (task) {
        if (badgeTask !== taskId) return;
        var name = String(task.project_name || '');
        for (var i = 0; i < STAGES.length; i += 1) {
          if (name.indexOf(STAGES[i].prefix) === 0) {
            badgeStage = STAGES[i];
            break;
          }
        }
        ensureBadge();
      })
      .catch(function () {
        // 조회 실패면 배지를 달지 않는다 (작업 화면 자체는 그대로 쓸 수 있다)
      });
  }

  function ensureBadge() {
    var taskId = jobTaskId();
    var old = document.querySelector('.dlp-stage');
    if (!taskId) {
      if (old) old.remove();
      badgeTask = null;
      return;
    }
    if (taskId !== badgeTask) {
      if (old) old.remove();
      lookupStage(taskId);
      return;
    }
    var host = document.querySelector('.cvat-left-header');
    if (!badgeStage || !host || old) return;
    var el = document.createElement('div');
    el.className = 'dlp-stage';
    var title = document.createElement('span');
    title.className = 'dlp-stage-title';
    title.textContent = badgeStage.title;
    var pill = document.createElement('span');
    pill.className = 'dlp-stage-pill dlp-stage-' + badgeStage.kind;
    pill.textContent = badgeStage.text;
    el.appendChild(title);
    el.appendChild(pill);
    var logo = host.querySelector('.cvat-logo-icon');
    host.insertBefore(el, logo ? logo.nextSibling : host.firstChild);
  }

  var scheduled = false;
  function refresh() {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(function () {
      scheduled = false;
      ensureToggle();
      ensureBadge();
    });
  }

  function start() {
    refresh();
    new MutationObserver(refresh).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
