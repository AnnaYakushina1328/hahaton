import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Modal,
  Pressable,
  RefreshControl,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

const API_BASE = "http://172.21.103.189:8080";

const riskMeta = {
  low: { title: "Низкий", color: "#6FBF8F", bg: "#ECFDF3" },
  medium: { title: "Средний", color: "#D99A3D", bg: "#FFF7E8" },
  high: { title: "Высокий", color: "#D36B6B", bg: "#FFF1F1" },
  unknown: { title: "Не рассчитан", color: "#8FA2B7", bg: "#F1F5F9" },
};

function normalizeRisk(value) {
  const risk = String(value || "").toLowerCase().trim();

  if (["high", "высокий", "красный"].includes(risk)) return "high";
  if (["medium", "средний", "желтый", "жёлтый"].includes(risk)) return "medium";
  if (["low", "низкий", "зеленый", "зелёный"].includes(risk)) return "low";

  return "unknown";
}

function normalizeHistoryResponse(raw) {
  if (Array.isArray(raw)) return raw;
  if (Array.isArray(raw?.events)) return raw.events;
  if (Array.isArray(raw?.history)) return raw.history;
  return [];
}

function getQueue(issue) {
  const value = String(issue || "").toUpperCase();
  return value.includes("-") ? value.split("-")[0] : value || "UNKNOWN";
}

function formatScore(score) {
  const number = Number(score);
  return Number.isNaN(number) ? "—" : number.toFixed(3);
}

function formatTime(value) {
  if (!value) return "—";

  const raw = String(value);
  const date = new Date(raw);

  if (Number.isNaN(date.getTime())) {
    return raw;
  }

  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function App() {
  const [events, setEvents] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [selectedRisk, setSelectedRisk] = useState("all");
  const [selectedQueue, setSelectedQueue] = useState("all");
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [lastUpdated, setLastUpdated] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  async function loadData({ silent = false } = {}) {
    try {
      if (!silent) setLoading(true);
      setError("");

      const [historyResponse, analyticsResponse] = await Promise.all([
        fetch(`${API_BASE}/api/history`),
        fetch(`${API_BASE}/api/analytics`),
      ]);

      if (!historyResponse.ok || !analyticsResponse.ok) {
        throw new Error("Backend вернул ошибку");
      }

      const historyJson = await historyResponse.json();
      const analyticsJson = await analyticsResponse.json();

      setEvents(normalizeHistoryResponse(historyJson));
      setAnalytics(analyticsJson);
      setLastUpdated(
        new Date().toLocaleTimeString("ru-RU", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
      );
    } catch (loadError) {
      setError(
        "Не удалось подключиться к backend. Проверь IP, Wi-Fi и запущен ли tracker_webhook.py."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadData();

    const timer = setInterval(() => {
      loadData({ silent: true });
    }, 30000);

    return () => clearInterval(timer);
  }, []);

  const queues = useMemo(() => {
    const result = new Set();

    events.forEach((event) => {
      result.add(getQueue(event.issue));
    });

    return Array.from(result).sort();
  }, [events]);

  const stats = useMemo(() => {
    const result = { total: events.length, high: 0, medium: 0, low: 0, unknown: 0 };

    events.forEach((event) => {
      const risk = normalizeRisk(event.risk_level);
      result[risk] += 1;
    });

    return result;
  }, [events]);

  const filteredEvents = useMemo(() => {
    return [...events]
      .sort((left, right) => String(right.time || "").localeCompare(String(left.time || "")))
      .filter((event) => {
        const risk = normalizeRisk(event.risk_level);
        const queue = getQueue(event.issue);

        if (selectedRisk !== "all" && risk !== selectedRisk) {
          return false;
        }

        if (selectedQueue !== "all" && queue !== selectedQueue) {
          return false;
        }

        return true;
      });
  }, [events, selectedRisk, selectedQueue]);

  const focusEvents = useMemo(() => {
    return events.filter((event) => {
      const risk = normalizeRisk(event.risk_level);
      return risk === "high" || risk === "medium";
    }).length;
  }, [events]);

  function onRefresh() {
    setRefreshing(true);
    loadData({ silent: true });
  }

  function resetFilters() {
    setSelectedRisk("all");
    setSelectedQueue("all");
  }

  const listTitle = selectedRisk === "all"
    ? "Задачи по фильтру"
    : `${riskMeta[selectedRisk]?.title || "Выбранный"} риск`;

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" />

      <ScrollView
        style={styles.page}
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      >
        <View style={styles.header}>
          <View>
            <Text style={styles.title}>Risk Tracker</Text>
            <Text style={styles.subtitle}>Мобильный контроль задач с риском срыва</Text>
          </View>

          <TouchableOpacity style={styles.refreshButton} onPress={() => loadData({ silent: true })}>
            <Text style={styles.refreshButtonText}>Обновить</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.statusLine}>
          <Text style={styles.statusText}>API: {API_BASE}</Text>
          <Text style={styles.statusText}>Обновлено: {lastUpdated || "—"}</Text>
        </View>

        {loading ? (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" />
            <Text style={styles.loadingText}>Загружаем данные...</Text>
          </View>
        ) : null}

        {error ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        ) : null}

        <View style={styles.heroCard}>
          <Text style={styles.heroLabel}>В фокусе тимлида</Text>
          <Text style={styles.heroValue}>{focusEvents}</Text>
          <Text style={styles.heroText}>
            задач со средним и высоким риском требуют внимания
          </Text>
        </View>

        <View style={styles.statsGrid}>
          <StatCard label="Всего событий" value={stats.total} />
          <StatCard label="Высокий риск" value={stats.high} risk="high" />
          <StatCard label="Средний риск" value={stats.medium} risk="medium" />
          <StatCard label="Низкий риск" value={stats.low} risk="low" />
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Фильтры</Text>

            <TouchableOpacity onPress={resetFilters}>
              <Text style={styles.resetText}>Сбросить</Text>
            </TouchableOpacity>
          </View>

          <Text style={styles.filterLabel}>Риск</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <RiskChip title="Все" active={selectedRisk === "all"} onPress={() => setSelectedRisk("all")} />
            <RiskChip title="Высокий" risk="high" active={selectedRisk === "high"} onPress={() => setSelectedRisk("high")} />
            <RiskChip title="Средний" risk="medium" active={selectedRisk === "medium"} onPress={() => setSelectedRisk("medium")} />
            <RiskChip title="Низкий" risk="low" active={selectedRisk === "low"} onPress={() => setSelectedRisk("low")} />
            <RiskChip title="Не рассчитан" risk="unknown" active={selectedRisk === "unknown"} onPress={() => setSelectedRisk("unknown")} />
          </ScrollView>

          <Text style={styles.filterLabel}>Очередь</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <QueueChip title="Все" active={selectedQueue === "all"} onPress={() => setSelectedQueue("all")} />
            {queues.map((queue) => (
              <QueueChip
                key={queue}
                title={queue}
                active={selectedQueue === queue}
                onPress={() => setSelectedQueue(queue)}
              />
            ))}
          </ScrollView>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <View>
              <Text style={styles.sectionTitle}>{listTitle}</Text>
              <Text style={styles.sectionHint}>Показано: {filteredEvents.length}</Text>
            </View>
          </View>

          {filteredEvents.length === 0 ? (
            <View style={styles.emptyBox}>
              <Text style={styles.emptyText}>Под выбранные фильтры задач нет</Text>
            </View>
          ) : (
            filteredEvents.slice(0, 30).map((event, index) => (
              <TaskCard
                key={`${event.issue}-${event.time}-${index}`}
                event={event}
                onPress={() => setSelectedEvent(event)}
              />
            ))
          )}
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Срез по очередям</Text>
          <Text style={styles.sectionHint}>Краткая сводка по проектным очередям</Text>

          <View style={styles.queueList}>
            {Object.entries(analytics?.risk_by_project || {}).map(([queue, bucket]) => {
              const high = bucket.high || 0;
              const medium = bucket.medium || 0;
              const low = bucket.low || 0;
              const unknown = bucket.unknown || 0;

              return (
                <View key={queue} style={styles.queueRow}>
                  <Text style={styles.queueName}>{queue}</Text>
                  <Text style={styles.queueText}>
                    высокий: {high} · средний: {medium} · низкий: {low} · не рассчитан: {unknown}
                  </Text>
                </View>
              );
            })}
          </View>
        </View>
      </ScrollView>

      <TaskDetailsModal event={selectedEvent} onClose={() => setSelectedEvent(null)} />
    </SafeAreaView>
  );
}

function StatCard({ label, value, risk }) {
  const meta = riskMeta[risk] || { color: "#2563EB", bg: "#EFF6FF" };

  return (
    <View style={[styles.statCard, { backgroundColor: meta.bg }]}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={[styles.statValue, { color: meta.color }]}>{value}</Text>
    </View>
  );
}

function RiskChip({ title, risk, active, onPress }) {
  const meta = riskMeta[risk] || { color: "#2563EB", bg: "#EFF6FF" };

  return (
    <TouchableOpacity
      onPress={onPress}
      style={[
        styles.chip,
        {
          backgroundColor: active ? meta.color : "#FFFFFF",
          borderColor: meta.color,
        },
      ]}
    >
      <Text style={[styles.chipText, { color: active ? "#FFFFFF" : "#172033" }]}>
        {title}
      </Text>
    </TouchableOpacity>
  );
}

function QueueChip({ title, active, onPress }) {
  return (
    <TouchableOpacity
      onPress={onPress}
      style={[
        styles.queueChip,
        {
          backgroundColor: active ? "#172033" : "#FFFFFF",
          borderColor: active ? "#172033" : "#CBD5E1",
        },
      ]}
    >
      <Text style={[styles.chipText, { color: active ? "#FFFFFF" : "#172033" }]}>
        {title}
      </Text>
    </TouchableOpacity>
  );
}

function TaskCard({ event, onPress }) {
  const risk = normalizeRisk(event.risk_level);
  const meta = riskMeta[risk];
  const queue = getQueue(event.issue);

  return (
    <TouchableOpacity onPress={onPress} style={styles.taskCard}>
      <View style={styles.taskTop}>
        <View>
          <Text style={styles.taskIssue}>{event.issue || "Без номера"}</Text>
          <Text style={styles.taskQueue}>{queue}</Text>
        </View>

        <View style={[styles.riskBadge, { backgroundColor: meta.bg }]}>
          <Text style={[styles.riskBadgeText, { color: meta.color }]}>
            {meta.title}
          </Text>
        </View>
      </View>

      <Text style={styles.taskTitle}>{event.title || "Без названия"}</Text>

      <Text style={styles.taskMeta}>
        Статус: {event.status || "—"} · Исполнитель: {event.assignee || "unknown"}
      </Text>

      <Text style={styles.taskMeta}>
        score {formatScore(event.risk_score)} · {formatTime(event.time)}
      </Text>
    </TouchableOpacity>
  );
}

function TaskDetailsModal({ event, onClose }) {
  if (!event) return null;

  const risk = normalizeRisk(event.risk_level);
  const meta = riskMeta[risk];
  const analysis = event.llm_analysis || {};

  return (
    <Modal visible={!!event} animationType="slide" transparent>
      <View style={styles.modalOverlay}>
        <View style={styles.modalCard}>
          <View style={styles.modalHeader}>
            <View style={styles.modalTitleBox}>
              <Text style={styles.modalIssue}>{event.issue}</Text>
              <Text style={styles.modalTitle}>{event.title || "Без названия"}</Text>
            </View>

            <Pressable onPress={onClose} style={styles.closeButton}>
              <Text style={styles.closeText}>×</Text>
            </Pressable>
          </View>

          <ScrollView showsVerticalScrollIndicator={false}>
            <View style={[styles.riskBox, { backgroundColor: meta.bg }]}>
              <Text style={[styles.riskBoxText, { color: meta.color }]}>
                {meta.title} риск · score {formatScore(event.risk_score)}
              </Text>
            </View>

            <Detail label="Статус" value={event.status || "—"} />
            <Detail label="Исполнитель" value={event.assignee || "unknown"} />
            <Detail label="Очередь" value={getQueue(event.issue)} />
            <Detail label="Время события" value={formatTime(event.time)} />

            <Text style={styles.detailLabel}>LLM-анализ</Text>
            <Text style={styles.detailText}>Понятность: {analysis.clarity_score || "—"}</Text>
            <Text style={styles.detailText}>Качество описания: {analysis.description_quality || "—"}</Text>
            <Text style={styles.detailText}>
              {analysis.clarity_comment || "LLM-анализ для этой задачи отсутствует."}
            </Text>

            <Text style={styles.detailLabel}>Причины риска</Text>
            {(analysis.risk_reasons || []).length ? (
              analysis.risk_reasons.map((item, index) => (
                <Text key={index} style={styles.detailText}>• {item}</Text>
              ))
            ) : (
              <Text style={styles.detailText}>Причины не указаны</Text>
            )}

            <Text style={styles.detailLabel}>Рекомендации</Text>
            {(analysis.recommendations || []).length ? (
              analysis.recommendations.map((item, index) => (
                <Text key={index} style={styles.detailText}>• {item}</Text>
              ))
            ) : (
              <Text style={styles.detailText}>Рекомендаций нет</Text>
            )}

            <Text style={styles.detailLabel}>Быстрое действие</Text>
            <Text style={styles.detailText}>
              {analysis.quick_fix || "Быстрое действие не указано"}
            </Text>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
}

function Detail({ label, value }) {
  return (
    <>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.detailText}>{value}</Text>
    </>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#F4F7FB" },
  page: { flex: 1 },
  content: { padding: 18, paddingBottom: 42 },

  header: {
    marginBottom: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 12,
  },
  title: { fontSize: 32, fontWeight: "900", color: "#111827" },
  subtitle: { marginTop: 4, fontSize: 15, color: "#64748B", fontWeight: "600" },

  refreshButton: {
    backgroundColor: "#172033",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 14,
  },
  refreshButtonText: { color: "#FFFFFF", fontWeight: "900" },

  statusLine: {
    marginBottom: 14,
    padding: 12,
    borderRadius: 16,
    backgroundColor: "#EAF0F8",
  },
  statusText: { color: "#52617A", fontWeight: "700", fontSize: 12 },

  loadingBox: { padding: 18, alignItems: "center" },
  loadingText: { marginTop: 8, color: "#64748B" },

  errorBox: {
    padding: 14,
    borderRadius: 18,
    backgroundColor: "#FFF1F1",
    borderWidth: 1,
    borderColor: "#FECACA",
    marginBottom: 14,
  },
  errorText: { color: "#991B1B", fontWeight: "800" },

  heroCard: {
    padding: 18,
    borderRadius: 24,
    backgroundColor: "#172033",
    marginBottom: 14,
  },
  heroLabel: { color: "#CBD5E1", fontWeight: "800" },
  heroValue: { marginTop: 4, color: "#FFFFFF", fontWeight: "900", fontSize: 42 },
  heroText: { color: "#CBD5E1", fontWeight: "700" },

  statsGrid: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  statCard: {
    width: "47.8%",
    padding: 16,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: "#E2E8F0",
  },
  statLabel: { color: "#52617A", fontWeight: "700", fontSize: 13 },
  statValue: { marginTop: 8, fontSize: 30, fontWeight: "900" },

  section: { marginTop: 22 },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  sectionTitle: { fontSize: 22, fontWeight: "900", color: "#111827" },
  sectionHint: { marginTop: 4, marginBottom: 10, color: "#64748B", fontWeight: "600" },
  resetText: { color: "#2563EB", fontWeight: "900" },
  filterLabel: { marginTop: 12, color: "#52617A", fontWeight: "900" },

  chip: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 999,
    borderWidth: 1,
    marginRight: 8,
    marginTop: 10,
  },
  queueChip: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 999,
    borderWidth: 1,
    marginRight: 8,
    marginTop: 10,
  },
  chipText: { fontWeight: "800" },

  taskCard: {
    padding: 16,
    borderRadius: 22,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E2E8F0",
    marginTop: 10,
  },
  taskTop: { flexDirection: "row", justifyContent: "space-between", gap: 10 },
  taskIssue: { fontSize: 18, fontWeight: "900", color: "#111827" },
  taskQueue: { marginTop: 2, color: "#64748B", fontWeight: "700" },
  taskTitle: { marginTop: 12, fontSize: 16, fontWeight: "800", color: "#172033" },
  taskMeta: { marginTop: 6, color: "#64748B", fontWeight: "600" },

  riskBadge: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    alignSelf: "flex-start",
  },
  riskBadgeText: { fontWeight: "900" },

  queueList: { marginTop: 10, gap: 10 },
  queueRow: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E2E8F0",
    borderRadius: 16,
    padding: 14,
  },
  queueName: { fontSize: 16, fontWeight: "900", color: "#111827" },
  queueText: { marginTop: 4, color: "#64748B", fontWeight: "700" },

  emptyBox: {
    marginTop: 12,
    padding: 18,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "#E2E8F0",
    backgroundColor: "#FFFFFF",
  },
  emptyText: { color: "#64748B", fontWeight: "700", textAlign: "center" },

  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(15, 23, 42, 0.45)",
    justifyContent: "flex-end",
  },
  modalCard: {
    maxHeight: "88%",
    backgroundColor: "#FFFFFF",
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    padding: 18,
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
    marginBottom: 14,
  },
  modalTitleBox: { flex: 1 },
  modalIssue: { fontSize: 20, fontWeight: "900", color: "#111827" },
  modalTitle: { marginTop: 4, color: "#64748B", fontWeight: "700" },

  closeButton: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: "#F1F5F9",
    alignItems: "center",
    justifyContent: "center",
  },
  closeText: { fontSize: 26, fontWeight: "700", color: "#334155" },

  riskBox: { padding: 14, borderRadius: 18, marginBottom: 14 },
  riskBoxText: { fontWeight: "900", fontSize: 16 },

  detailLabel: { marginTop: 14, marginBottom: 4, fontWeight: "900", color: "#111827" },
  detailText: { color: "#475569", lineHeight: 22, fontWeight: "600" },
});
