import { useState } from 'react';
import { ActivityIndicator, Image, Pressable, SafeAreaView, ScrollView, StyleSheet, Switch, Text, TextInput, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { File } from 'expo-file-system';
import { StatusBar } from 'expo-status-bar';

const API_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';
const API_KEY = process.env.EXPO_PUBLIC_API_KEY;

type Finding = { name: string; confidence: number; threshold: number };
type Result = {
  assessment_id: string;
  findings: Finding[];
  assessment: {
    severity: string;
    urgency: string;
    guidance: string;
    cost_band: string;
    indicative_cost_aed: string;
    requires_human_review: boolean;
  };
};
type Partner = { id: string; name: string; city: string; phone?: string; address?: string };

function createImageUpload(image: ImagePicker.ImagePickerAsset) {
  const body = new FormData();
  const file = new File(image.uri);
  body.append('file', file as unknown as Blob);
  return body;
}

export default function App() {
  const [image, setImage] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [correction, setCorrection] = useState('');
  const [consent, setConsent] = useState(false);
  const [partners, setPartners] = useState<Partner[]>([]);
  const [partnerId, setPartnerId] = useState('');
  const [contactName, setContactName] = useState('');
  const [contactPhone, setContactPhone] = useState('');
  const [preferredTime, setPreferredTime] = useState('');
  const [locationConsent, setLocationConsent] = useState(false);
  const [bookingSent, setBookingSent] = useState(false);

  async function chooseImage() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Photo access is required to analyze an image.');
      return;
    }
    const selected = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.9 });
    if (!selected.canceled) {
      setImage(selected.assets[0]);
      setResult(null);
      setError('');
      setFeedbackSent(false);
      setCorrection('');
      setConsent(false);
      setPartners([]);
      setPartnerId('');
      setBookingSent(false);
    }
  }

  async function sendFeedback() {
    if (!image || !result || !consent || !correction.trim()) return;
    const body = createImageUpload(image);
    body.append('corrected_labels', correction.trim());
    body.append('consent_to_training', String(consent));
    try {
      const response = await fetch(`${API_URL}/v1/feedback`, { method: 'POST', headers: API_KEY ? { 'X-API-Key': API_KEY } : undefined, body });
      if (!response.ok) throw new Error('Feedback could not be submitted.');
      setFeedbackSent(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Feedback could not be submitted.');
    }
  }

  async function takePhoto() {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setError('Camera access is required to take a vehicle photo.');
      return;
    }
    const captured = await ImagePicker.launchCameraAsync({ quality: 0.9 });
    if (!captured.canceled) {
      setImage(captured.assets[0]);
      setResult(null);
      setError('');
    }
  }

  async function analyze() {
    if (!image) return;
    setBusy(true);
    setError('');
    try {
      const body = createImageUpload(image);
      const response = await fetch(`${API_URL}/v1/analyze`, { method: 'POST', headers: API_KEY ? { 'X-API-Key': API_KEY } : undefined, body });
      if (!response.ok) throw new Error((await response.json()).detail ?? 'Analysis failed.');
      setResult(await response.json());
      const partnerResponse = await fetch(`${API_URL}/v1/partners?city=Dubai`);
      if (partnerResponse.ok) {
        const partnerData = await partnerResponse.json();
        setPartners(partnerData.partners ?? []);
        setPartnerId(partnerData.partners?.[0]?.id ?? '');
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Analysis failed.');
    } finally {
      setBusy(false);
    }
  }

  async function requestBooking() {
    if (!result || !partnerId || !contactName.trim() || !contactPhone.trim() || !preferredTime.trim() || !locationConsent) return;
    const body = new URLSearchParams({
      partner_id: partnerId,
      contact_name: contactName.trim(),
      contact_phone: contactPhone.trim(),
      preferred_time: preferredTime.trim(),
      city: 'Dubai',
      location_consent: 'true',
      assessment_id: result.assessment_id,
    });
    try {
      const response = await fetch(`${API_URL}/v1/booking-requests`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded', ...(API_KEY ? { 'X-API-Key': API_KEY } : {}) }, body: body.toString() });
      if (!response.ok) throw new Error((await response.json()).detail ?? 'Booking request failed.');
      setBookingSent(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Booking request failed.');
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.hero}>
          <Text style={styles.eyebrow}>AUTOTRIAGE</Text>
          <Text style={styles.title}>A calmer first look at vehicle damage.</Text>
          <Text style={styles.subtitle}>Private by default. Designed to help you decide what deserves a closer look.</Text>
        </View>
        <View style={styles.sheet}>
          {image ? <Image source={{ uri: image.uri }} style={styles.preview} /> : <View style={styles.empty}><Text style={styles.emptyTitle}>No photo yet</Text><Text style={styles.emptyText}>Choose a clear exterior photo with the affected area visible.</Text></View>}
          <View style={styles.actionRow}>
            <Pressable style={[styles.primary, styles.secondaryAction]} onPress={takePhoto}><Text style={styles.secondaryText}>Take photo</Text></Pressable>
            <Pressable style={[styles.primary, styles.secondaryAction]} onPress={chooseImage}><Text style={styles.secondaryText}>Choose photo</Text></Pressable>
          </View>
          {image && <Pressable style={[styles.primary, busy && styles.disabled]} onPress={analyze} disabled={busy}>{busy ? <ActivityIndicator color="#fff" /> : <Text style={styles.primaryText}>Analyze photo</Text>}</Pressable>}
          {error ? <Text style={styles.error}>{error}</Text> : null}
        </View>
        {result ? <View style={styles.result}>
          <Text style={styles.resultLabel}>ASSESSMENT</Text>
          <Text style={styles.resultTitle}>{result.assessment.urgency}</Text>
          <Text style={styles.guidance}>{result.assessment.guidance}</Text>
          <View style={styles.metaRow}><Text style={styles.metaKey}>Severity</Text><Text style={styles.metaValue}>{result.assessment.severity}</Text></View>
          <View style={styles.metaRow}><Text style={styles.metaKey}>Indicative cost</Text><Text style={styles.metaValue}>{result.assessment.cost_band}</Text></View>
          <View style={styles.metaRow}><Text style={styles.metaKey}>Dubai range</Text><Text style={styles.metaValue}>{result.assessment.indicative_cost_aed}</Text></View>
          <Text style={styles.resultLabel}>FINDINGS</Text>
          {result.findings.length ? result.findings.map((finding) => <View style={styles.finding} key={finding.name}><Text style={styles.findingName}>{finding.name}</Text><Text style={styles.findingScore}>{Math.round(finding.confidence * 100)}%</Text></View>) : <Text style={styles.guidance}>No confident damage class was found.</Text>}
          {result.assessment.requires_human_review ? <Text style={styles.review}>Professional review recommended.</Text> : null}
          <TextInput style={styles.feedbackInput} value={correction} onChangeText={setCorrection} placeholder="Correct labels, or enter none" placeholderTextColor="#a9c8b6" editable={!feedbackSent} />
          <View style={styles.consentRow}><Switch value={consent} onValueChange={setConsent} disabled={feedbackSent} /><Text style={styles.feedbackText}>I consent to storing this image and correction for model improvement.</Text></View>
          <Pressable style={[styles.feedback, (!consent || !correction.trim() || feedbackSent) && styles.disabled]} onPress={sendFeedback} disabled={!consent || !correction.trim() || feedbackSent}>
            <Text style={styles.feedbackText}>{feedbackSent ? 'Correction submitted for review' : 'Submit reviewed correction'}</Text>
          </Pressable>
          <Text style={styles.resultLabel}>DUBAI INSPECTION</Text>
          {partners.length ? <>
            <Text style={styles.guidance}>Request an inspection from a verified partner. This sends a request only; the shop must confirm availability.</Text>
            <View style={styles.partnerList}>{partners.map((partner) => <Pressable key={partner.id} style={[styles.partner, partnerId === partner.id && styles.partnerSelected]} onPress={() => setPartnerId(partner.id)}><Text style={styles.partnerName}>{partner.name}</Text><Text style={styles.partnerDetail}>{partner.address ?? partner.city}</Text></Pressable>)}</View>
            <TextInput style={styles.feedbackInput} value={contactName} onChangeText={setContactName} placeholder="Your name" placeholderTextColor="#a9c8b6" />
            <TextInput style={styles.feedbackInput} value={contactPhone} onChangeText={setContactPhone} placeholder="Phone number" placeholderTextColor="#a9c8b6" keyboardType="phone-pad" />
            <TextInput style={styles.feedbackInput} value={preferredTime} onChangeText={setPreferredTime} placeholder="Preferred inspection time" placeholderTextColor="#a9c8b6" />
            <View style={styles.consentRow}><Switch value={locationConsent} onValueChange={setLocationConsent} disabled={bookingSent} /><Text style={styles.feedbackText}>I consent to sharing my contact details and location request with this partner.</Text></View>
            <Pressable style={[styles.feedback, (!locationConsent || !contactName.trim() || !contactPhone.trim() || !preferredTime.trim() || bookingSent) && styles.disabled]} onPress={requestBooking} disabled={!locationConsent || !contactName.trim() || !contactPhone.trim() || !preferredTime.trim() || bookingSent}><Text style={styles.feedbackText}>{bookingSent ? 'Inspection request sent' : 'Request inspection'}</Text></Pressable>
          </> : <Text style={styles.guidance}>No verified Dubai partners are configured yet. We will not show unverified businesses.</Text>}
        </View> : null}
        <Text style={styles.disclaimer}>AutoTriage is a screening aid, not a mechanic, repair quotation, insurance assessor, or guarantee that a vehicle is safe to drive.</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#12383b' },
  container: { padding: 20, gap: 18, backgroundColor: '#edf2e9', minHeight: '100%' },
  hero: { paddingTop: 28, paddingBottom: 26 },
  eyebrow: { color: '#f1b66b', fontWeight: '800', letterSpacing: 2, fontSize: 12 },
  title: { color: '#fff', fontSize: 36, lineHeight: 40, fontWeight: '800', marginTop: 12 },
  subtitle: { color: '#d2e5d9', fontSize: 16, lineHeight: 23, marginTop: 12 },
  sheet: { backgroundColor: '#fffdf8', borderRadius: 16, padding: 16, gap: 12 },
  empty: { height: 210, borderRadius: 12, borderWidth: 1, borderColor: '#cbd8cc', borderStyle: 'dashed', alignItems: 'center', justifyContent: 'center', padding: 24 },
  emptyTitle: { color: '#173b3f', fontSize: 20, fontWeight: '700' },
  emptyText: { color: '#617067', textAlign: 'center', marginTop: 8, lineHeight: 20 },
  preview: { height: 260, borderRadius: 12, backgroundColor: '#dbe5dc' },
  primary: { minHeight: 52, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: '#e07a5f' },
  primaryText: { color: '#fff', fontSize: 16, fontWeight: '800' },
  actionRow: { flexDirection: 'row', gap: 10 },
  secondaryAction: { flex: 1, backgroundColor: '#dce8df' },
  secondaryText: { color: '#173b3f', fontSize: 15, fontWeight: '800' },
  disabled: { opacity: 0.6 },
  error: { color: '#b23a48', lineHeight: 20 },
  result: { backgroundColor: '#12383b', borderRadius: 16, padding: 20 },
  resultLabel: { color: '#f1b66b', fontSize: 11, letterSpacing: 1.8, fontWeight: '800', marginTop: 4 },
  resultTitle: { color: '#fff', fontSize: 27, fontWeight: '800', marginTop: 8, textTransform: 'capitalize' },
  guidance: { color: '#d8e9dc', fontSize: 15, lineHeight: 22, marginTop: 10 },
  metaRow: { borderTopWidth: 1, borderTopColor: '#2e5a5b', marginTop: 16, paddingTop: 12, flexDirection: 'row', justifyContent: 'space-between', gap: 12 },
  metaKey: { color: '#a9c8b6' },
  metaValue: { color: '#fff', fontWeight: '700', flexShrink: 1, textAlign: 'right' },
  finding: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#2e5a5b' },
  findingName: { color: '#fff', fontSize: 16, textTransform: 'capitalize' },
  findingScore: { color: '#f1b66b', fontWeight: '800' },
  review: { color: '#f1b66b', fontWeight: '700', marginTop: 16 },
  feedback: { marginTop: 18, borderWidth: 1, borderColor: '#6f9b8b', borderRadius: 10, padding: 13, alignItems: 'center' },
  feedbackText: { color: '#d8e9dc', fontWeight: '700', textAlign: 'center' },
  feedbackInput: { marginTop: 14, borderWidth: 1, borderColor: '#6f9b8b', borderRadius: 10, padding: 12, color: '#fff' },
  consentRow: { marginTop: 12, flexDirection: 'row', alignItems: 'center', gap: 8 },
  partnerList: { gap: 8, marginTop: 12 },
  partner: { borderWidth: 1, borderColor: '#42716b', borderRadius: 10, padding: 12 },
  partnerSelected: { borderColor: '#f1b66b', backgroundColor: '#1c4b4b' },
  partnerName: { color: '#fff', fontWeight: '800' },
  partnerDetail: { color: '#a9c8b6', marginTop: 4 },
  disclaimer: { color: '#617067', fontSize: 12, lineHeight: 18, paddingBottom: 24 },
});
